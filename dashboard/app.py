"""
Highway Guardian AI -- Live Monitoring Dashboard.

Run with:
    streamlit run dashboard/app.py

Shows recent incidents, their severity, dispatch status, snapshot evidence,
and a map of where accidents have been detected.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow running via `streamlit run dashboard/app.py` from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from database import incident_db  # noqa: E402

st.set_page_config(page_title="Highway Guardian AI", layout="wide", page_icon="🚨")

st.title("🚨 Highway Guardian AI — Live Monitoring Dashboard")
st.caption("Real-time highway accident detection and automated emergency response")

incidents = incident_db.get_recent_incidents(limit=100)

if not incidents:
    st.info("No incidents logged yet. Once the pipeline confirms an accident, it will appear here automatically.")
    st.stop()

# ---------------- Summary metrics ----------------
severity_counts = {"Minor Impact": 0, "Substantial Impact": 0, "Critical Impact": 0}
for inc in incidents:
    severity_counts[inc.severity] = severity_counts.get(inc.severity, 0) + 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Incidents (recent)", len(incidents))
col2.metric("Minor", severity_counts.get("Minor Impact", 0))
col3.metric("Substantial", severity_counts.get("Substantial Impact", 0))
col4.metric("Critical", severity_counts.get("Critical Impact", 0))

st.divider()

# ---------------- Map ----------------
st.subheader("Incident Map")
map_df = pd.DataFrame([{"lat": inc.lat, "lon": inc.lon} for inc in incidents if inc.lat and inc.lon])
if not map_df.empty:
    st.map(map_df, size=40)
else:
    st.write("No geolocated incidents yet.")

st.divider()

# ---------------- Incident table ----------------
st.subheader("Recent Incidents")

severity_filter = st.multiselect(
    "Filter by severity", options=list(severity_counts.keys()), default=list(severity_counts.keys())
)

for inc in incidents:
    if inc.severity not in severity_filter:
        continue

    severity_color = {
        "Minor Impact": "🟡",
        "Substantial Impact": "🟠",
        "Critical Impact": "🔴",
    }.get(inc.severity, "⚪")

    with st.expander(
        f"{severity_color} {inc.timestamp} — {inc.camera_id} — {inc.severity} "
        f"({inc.vehicle_count} vehicles) — {inc.address[:60]}..."
    ):
        c1, c2 = st.columns([1, 1])

        with c1:
            if inc.snapshot_path and Path(inc.snapshot_path).exists():
                st.image(inc.snapshot_path, caption="Incident snapshot", use_column_width=True)
            else:
                st.write("No snapshot available.")

        with c2:
            st.write(f"**Address:** {inc.address}")
            st.write(f"**Coordinates:** {inc.lat}, {inc.lon}")
            st.write(f"**Classifier confidence:** {inc.classifier_confidence:.2f}")
            st.write(f"**Severity confidence:** {inc.severity_confidence:.2f}")
            st.write(f"**Vehicles involved:** {', '.join(json.loads(inc.vehicle_classes))}")
            st.write(f"**Status:** {inc.status}")

            st.markdown("**Dispatch Log**")
            try:
                dispatch_log = json.loads(inc.dispatch_log)
                dispatch_df = pd.DataFrame(dispatch_log)
                st.dataframe(dispatch_df, use_container_width=True)
            except Exception:
                st.write("No dispatch log available.")

            police = json.loads(inc.police_contact) if inc.police_contact else {}
            hospital = json.loads(inc.hospital_contact) if inc.hospital_contact else {}
            st.write(f"**Police notified:** {police.get('name', 'N/A')} ({police.get('phone', 'N/A')})")
            st.write(f"**Hospital notified:** {hospital.get('name', 'N/A')} ({hospital.get('phone', 'N/A')})")

st.divider()
st.caption("Highway Guardian AI — fully automated detection-to-dispatch pipeline. "
           "Data refreshes each time this page is reloaded.")
