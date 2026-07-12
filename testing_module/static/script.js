// testing_module/static/script.js
// Handles: source-type field toggling, applying/stopping the video source,
// polling live stats + history, and popping the alert overlay the moment a
// NEW confirmed accident shows up in /api/status.

const sourceTypeRadios = document.querySelectorAll('input[name="source_type"]');
const fieldWebcam = document.getElementById('field-webcam');
const fieldVideoFile = document.getElementById('field-video_file');
const fieldIpCamera = document.getElementById('field-ip_camera');

const videoFeedImg = document.getElementById('video-feed');
const noFeedPlaceholder = document.getElementById('no-feed-placeholder');
const sourceStatusEl = document.getElementById('source-status');

const btnApply = document.getElementById('btn-apply');
const btnStop = document.getElementById('btn-stop');

const alertOverlay = document.getElementById('alert-overlay');
const alertDetails = document.getElementById('alert-details');
const alertScreenshot = document.getElementById('alert-screenshot');
const alertDismiss = document.getElementById('alert-dismiss');

let lastSeenAlertId = null;

// ---------------- Source type field toggling ----------------
function updateVisibleFields() {
  const selected = document.querySelector('input[name="source_type"]:checked').value;
  fieldWebcam.style.display = selected === 'webcam' ? 'block' : 'none';
  fieldVideoFile.style.display = selected === 'video_file' ? 'block' : 'none';
  fieldIpCamera.style.display = selected === 'ip_camera' ? 'block' : 'none';
}
sourceTypeRadios.forEach(r => r.addEventListener('change', updateVisibleFields));
updateVisibleFields();

// ---------------- Apply / Stop ----------------
btnApply.addEventListener('click', async () => {
  const sourceType = document.querySelector('input[name="source_type"]:checked').value;
  let sourceValue;
  if (sourceType === 'webcam') sourceValue = document.getElementById('webcam_index').value || '0';
  if (sourceType === 'video_file') sourceValue = document.getElementById('video_file_path').value;
  if (sourceType === 'ip_camera') sourceValue = document.getElementById('ip_camera_url').value;

  const simulateDispatch = document.getElementById('simulate_dispatch').checked;

  sourceStatusEl.textContent = 'Opening source...';

  const resp = await fetch('/api/source', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_type: sourceType, source_value: sourceValue, simulate_dispatch: simulateDispatch })
  });
  const data = await resp.json();

  if (data.ok) {
    sourceStatusEl.textContent = `Active source: ${data.source_label}`;
    noFeedPlaceholder.style.display = 'none';
    // Cache-bust so the browser opens a fresh MJPEG connection
    videoFeedImg.src = '/video_feed?t=' + Date.now();
  } else {
    sourceStatusEl.textContent = `Error: ${data.error || 'could not open source'}`;
  }
});

btnStop.addEventListener('click', async () => {
  await fetch('/api/stop', { method: 'POST' });
  videoFeedImg.src = '';
  noFeedPlaceholder.style.display = 'block';
  sourceStatusEl.textContent = 'Stopped. No source active.';
});

// ---------------- Live stats polling ----------------
async function pollStatus() {
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();

    document.getElementById('stat-fps').textContent = data.fps.toFixed(1);
    document.getElementById('stat-frames').textContent = data.frame_count;

    if (data.last_result) {
      document.getElementById('stat-vehicles').textContent = data.last_result.vehicle_count;
      document.getElementById('stat-conf').textContent = (data.last_result.classifier_conf * 100).toFixed(1) + '%';
    }

    if (data.running && data.source_label) {
      sourceStatusEl.textContent = `Active source: ${data.source_label}`;
    }

    // New confirmed alert? -> popup
    if (data.last_alert && data.last_alert.id !== lastSeenAlertId) {
      lastSeenAlertId = data.last_alert.id;
      showAlertPopup(data.last_alert);
    }
  } catch (e) {
    // Server not reachable yet / mid-restart -- ignore and retry next tick
  }
}
setInterval(pollStatus, 1000);

// ---------------- History table polling ----------------
async function pollHistory() {
  try {
    const resp = await fetch('/api/history?limit=50');
    const rows = await resp.json();
    const tbody = document.getElementById('history-body');

    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-row">No confirmed accidents logged yet in this session.</td></tr>';
      return;
    }

    tbody.innerHTML = rows.map(r => `
      <tr>
        <td>${r.timestamp}</td>
        <td>${r.source}</td>
        <td class="severity-${r.severity.split(' ')[0]}">${r.severity}</td>
        <td>${(r.severity_confidence * 100).toFixed(1)}%</td>
        <td>${(r.classifier_confidence * 100).toFixed(1)}%</td>
        <td>${r.vehicle_count}</td>
        <td><a href="/test_captures/${r.screenshot}" target="_blank">view</a></td>
      </tr>
    `).join('');
  } catch (e) {
    // ignore transient errors
  }
}
setInterval(pollHistory, 2000);
pollHistory();

// ---------------- Alert popup ----------------
function showAlertPopup(alert) {
  alertDetails.textContent =
    `${alert.severity} accident detected from source "${alert.source}" ` +
    `(severity confidence ${(alert.severity_confidence * 100).toFixed(1)}%, ` +
    `${alert.vehicle_count} vehicles involved) at ${alert.timestamp}.`;
  alertScreenshot.src = `/test_captures/${alert.screenshot}`;
  alertOverlay.classList.remove('hidden');

  // Optional audible beep so a tester doesn't have to be looking at the screen
  try {
    const beep = new AudioContext();
    const osc = beep.createOscillator();
    osc.frequency.value = 880;
    osc.connect(beep.destination);
    osc.start();
    setTimeout(() => osc.stop(), 250);
  } catch (e) { /* AudioContext may be blocked until user interacts with the page -- non-critical */ }
}

alertDismiss.addEventListener('click', () => {
  alertOverlay.classList.add('hidden');
});
