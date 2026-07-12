# Highway Guardian AI
Real-Time Highway Accident Detection and Intelligent Emergency Response System

End-to-end pipeline: CCTV/RTSP feed → vehicle detection & tracking → accident
detection → verification → severity classification → location lookup →
**fully automated** police/hospital notification (voice call + SMS + email,
no human clicks anything) → live dashboard → incident database.

Dataset used: [Road Accidents from CCTV Footages Dataset](https://www.kaggle.com/datasets/suryaprabhakaran2005/road-accidents-from-cctv-footages-dataset)
(6,191 Accident images / 15,420 NonAccident images / severity-labeled subset).

---

## 1. Project layout

```
highway_guardian_ai/
├── config/
│   ├── settings.py                     # central config, loads .env
│   ├── cameras.json                    # per-camera GPS registry
│   └── local_emergency_directory.json  # fallback real phone numbers per region
├── data_prep/
│   ├── prepare_accident_dataset.py     # raw Kaggle -> YOLO-cls train/val folders
│   └── prepare_severity_dataset.py     # raw Kaggle severity folders -> train/val
├── training/
│   ├── train_accident_classifier.py    # trains Accident vs NonAccident model
│   └── train_severity_classifier.py    # trains Minor/Substantial/Critical model
├── detection/
│   ├── vehicle_detector.py             # YOLOv8 + ByteTrack (pretrained, COCO)
│   └── trajectory_tracker.py           # per-vehicle motion history & heuristics
├── accident/
│   ├── accident_classifier.py          # loads trained accident model
│   ├── severity_classifier.py          # loads trained severity model
│   └── verification.py                 # combines classifier + motion, temporal confirm
├── location/
│   └── geolocation.py                  # camera GPS -> address (OSM Nominatim)
├── emergency/
│   ├── service_finder.py               # nearest police/hospital (OSM Overpass + fallback)
│   └── notifier.py                     # Twilio auto call/SMS + email dispatch
├── database/
│   └── incident_db.py                  # SQLite incident log
├── pipeline/
│   └── main_pipeline.py                # orchestrates the full real-time flow
├── dashboard/
│   └── app.py                          # Streamlit live monitoring dashboard
├── utils/helpers.py                    # drawing/snapshot helpers
├── models/                             # trained weights land here
├── static/snapshots/                   # incident evidence photos
├── requirements.txt
└── .env.example                        # copy to .env and fill in real values
```

---

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: Twilio credentials, SMTP credentials, camera GPS, RTSP URL
```

---

## 3. Prepare the dataset

1. Download **Road Accidents from CCTV Footages Dataset** from Kaggle and unzip it, e.g. to `~/kaggle_data`.
2. Build the two training sets:

```bash
python data_prep/prepare_accident_dataset.py \
    --raw_root ~/kaggle_data \
    --out_dir  ~/datasets/dataset_accident \
    --val_split 0.15

python data_prep/prepare_severity_dataset.py \
    --raw_root ~/kaggle_data \
    --out_dir  ~/datasets/dataset_severity \
    --val_split 0.15
```

These scripts auto-detect the dataset's doubled folder names
(`Accident/Accident/...`) and the `SeverityScore/Severity Score Dataset/1|2|3`
layout, and produce a clean `train/<class>/*.jpg`, `val/<class>/*.jpg`
structure that Ultralytics classification training expects.

---

## 4. Train the two classifiers

```bash
python training/train_accident_classifier.py \
    --data ~/datasets/dataset_accident --epochs 30 --imgsz 224

python training/train_severity_classifier.py \
    --data ~/datasets/dataset_severity --epochs 40 --imgsz 224
```

Weights are written to `models/accident_classifier/weights/best.pt` and
`models/severity_classifier/weights/best.pt` — the exact paths
`config/settings.py` already points to, so no further wiring is needed.
Training on CPU works but is slow; a single free-tier GPU (e.g. Colab) trains
either model in well under an hour at these image counts.

---

## 5. Configure emergency contacts

Real phone numbers for specific police stations/hospitals are **not**
reliably available from any free public API — OpenStreetMap only has them
where a volunteer added the tag. So the system:

1. Queries OSM Overpass for the nearest station/hospital by GPS distance.
2. Uses its phone number if OSM has one tagged.
3. Otherwise falls back to `config/local_emergency_directory.json`, which
   **you must fill in with verified real numbers** for the highway stretch(es)
   your cameras cover.
4. If neither source has anything, it falls back to the national emergency
   number (`112` by default — change per country in the same JSON file) so a
   human dispatcher can route the call.

Edit `config/local_emergency_directory.json` and `config/cameras.json`
before deploying against a real camera.

---

## 6. Run the real-time pipeline

```bash
# Webcam / local test video
python -m pipeline.main_pipeline --source 0 --display
python -m pipeline.main_pipeline --source path/to/test_video.mp4 --display

# Real CCTV camera over RTSP
python -m pipeline.main_pipeline \
    --source "rtsp://user:pass@camera-ip:554/stream1" \
    --camera_id CAM_NH44_KM231 \
    --control_room_email controlroom@example.com
```

What happens automatically, with no person involved:

1. Frames are sampled from the stream (`FRAME_SAMPLE_RATE`, default every 5th frame).
2. YOLOv8 detects vehicles; ByteTrack assigns stable IDs across frames.
3. The trained classifier scores each sampled frame Accident / NonAccident.
4. The verification module requires the classifier **and** a motion
   signature (box overlap or sharp deceleration) to agree across
   `ACCIDENT_CONFIRMATION_FRAMES` consecutive frames (default 5) before
   calling it a confirmed incident — this is what keeps a stationary truck
   or camera glare from triggering a false emergency call.
5. On confirmation: severity is classified, the camera's GPS is
   reverse-geocoded to a street address, the nearest police station and
   hospital are located, and Twilio places an automated voice call
   (text-to-speech) **and** sends an SMS to both — plus an optional email to
   a control-room address. All of this fires immediately, without a
   dashboard operator approving anything.
6. The incident (snapshot, severity, confidences, contacts notified,
   per-channel delivery status) is written to `database/incidents.db`.
7. A `INCIDENT_COOLDOWN_SECONDS` window (default 300s) prevents the same
   ongoing incident from re-triggering repeated calls every few seconds.

---

## 7. Run the dashboard

```bash
streamlit run dashboard/app.py
```

Shows: live incident count by severity, a map of all detected incidents,
and an expandable record per incident with its snapshot, address, involved
vehicle classes, and full per-channel dispatch log (call/SMS/email —
sent/failed/skipped).

---

## 8. Notes, limits, and what to verify before real deployment

- **Twilio and SMTP credentials are required** for calls/SMS/email to
  actually send; without them the system logs what *would* have been sent
  (`status: skipped_no_credentials`) rather than pretending to succeed.
- **Verify local emergency numbers yourself.** OSM data quality varies by
  region; always confirm the numbers in `local_emergency_directory.json`
  are current and correct for your jurisdiction before relying on this for
  real dispatch. Consider keeping a human-monitored fallback channel
  (e.g. the dashboard/control-room email) even after automating the
  call/SMS path.
- **Model accuracy** depends entirely on how well the two classifiers train
  on your data — validate precision/recall on a held-out set before trusting
  it to dial emergency services unattended, and tune
  `ACCIDENT_CONF_THRESHOLD` / `ACCIDENT_CONFIRMATION_FRAMES` in
  `config/settings.py` against your own false-positive rate.
- **RTSP performance**: `FRAME_SAMPLE_RATE` trades detection latency for
  throughput; lower it for faster response, raise it if the pipeline can't
  keep up with the stream in real time on your hardware.
