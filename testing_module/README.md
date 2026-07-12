# Real-Time Camera Testing Module

A standalone Flask GUI for validating the **entire** Highway Guardian AI
pipeline — camera input → vehicle detection → accident classification →
temporal verification → severity classification → alerting — using a
webcam, a phone camera, or a recorded video, before pointing any of it at a
real highway CCTV feed.

It reuses your actual trained models and pipeline logic
(`detection/`, `accident/`, `location/`, `emergency/`) directly — this is a
real test of the pipeline, not a separate mock.

---

## 1. What it does NOT do

By default this module **never contacts real emergency services**. It only
detects, verifies, classifies, screenshots, logs, and alerts on-screen.

If you tick **"Simulate emergency dispatch"** in the GUI, it will run the
*real* `emergency/notifier.py` dispatch chain in dry-run — but Twilio/SMTP
calls only actually go out if you've put real credentials in `.env`; without
them, each channel just logs `skipped_no_credentials`, identical to
production behavior. Use this checkbox deliberately.

---

## 2. Run it

From the project root (not inside `testing_module/`):

```bash
pip install -r requirements.txt
python -m testing_module.app
```

Open **http://localhost:5001** in a browser. Model loading happens once at
startup and can take several seconds — wait for the "Testing module ready"
log line before opening the page.

---

## 3. Choosing an input source

The GUI has three source types, switchable at any time without restarting
the server:

### Webcam
Just set the device index (usually `0` for a laptop's built-in camera) and
click **Apply**.

### Video file
Enter a path to a video file **on the machine running this Flask app**
(not your browser's machine, if they're different) — e.g. a sample
accident clip you've saved for testing. Click **Apply**.

### IP Camera (DroidCam / IP Webcam / any phone camera app)
Use your phone as a stand-in for a highway CCTV camera:

- **IP Webcam** (Android, free): install it, tap "Start server", and it
  will show a URL like `http://192.168.1.50:8080/video` — enter that.
- **DroidCam**: install it, note the IP it shows, and use
  `http://192.168.1.50:4747/video`.
- Your phone and the machine running this app must be on the **same Wi-Fi
  network**.
- This same field works for a real CCTV camera's **RTSP URL**
  (`rtsp://user:pass@camera-ip:554/stream1`) — that's the intended
  migration path: swap the URL here first to sanity-check the stream, then
  point `pipeline/main_pipeline.py` at the same URL for production.

---

## 4. What you'll see

- **Live annotated video**: bounding boxes with vehicle class + confidence,
  an FPS/vehicle-count/confidence overlay, and a red banner across the top
  the moment an accident is confirmed (or "verifying" while it's building
  up confirmation frames but hasn't cleared the threshold yet).
- **Live Stats panel**: FPS, vehicle count, accident confidence, total
  frames processed — updated every second.
- **Automatic alert popup**: the instant an accident is *confirmed* (not on
  every noisy single-frame flicker), a modal pops up over the page with the
  severity, confidence, vehicle count, and the screenshot — plus a short
  beep. Click "Dismiss" to close it.
- **Detection History table**: every confirmed accident this session, with
  timestamp, source, severity, confidences, vehicle count, and a link to
  the saved screenshot.

---

## 5. Where things get saved

- Screenshots: `testing_module/test_captures/*.jpg` (kept separate from
  `static/snapshots/`, the production evidence folder).
- CSV logs: `testing_module/test_logs/test_session_<date>.csv` — one row
  per confirmed accident, with a full date/time stamp.

Neither of these touches `database/incidents.db`, the production incident
log — test runs and real deployments never mix.

---

## 6. Tuning false positives during testing

If you're seeing too many/few confirmations while testing, the same knobs
apply as in production — edit `config/settings.py` or your `.env`:

- `ACCIDENT_CONF_THRESHOLD` — minimum classifier confidence to count as an
  "accident" frame at all.
- `ACCIDENT_CONFIRMATION_FRAMES` — how many consecutive sampled frames must
  agree before the alert fires.

Because this module calls the exact same `AccidentVerifier` class as
`pipeline/main_pipeline.py`, whatever you tune here behaves identically once
you deploy to a real camera.

---

## 7. Moving to a real highway camera

Once you're satisfied the pipeline behaves correctly here:

1. Add the real camera's RTSP URL and GPS coordinates to
   `config/cameras.json`.
2. Fill in real, verified phone numbers in
   `config/local_emergency_directory.json`.
3. Run the production pipeline instead of this testing module:
   ```bash
   python -m pipeline.main_pipeline --source "rtsp://..." --camera_id CAM_ID
   ```

No detection/classification/verification code changes between testing and
production — only the input source and whether dispatch is real or
simulated.
