"""
Location Identification module.

Each camera has a fixed lat/lon (from config/cameras.json or .env). This
module turns that coordinate into a human-readable address via OpenStreetMap
Nominatim (free, no API key) so alerts/emails/SMS include a real address,
not just raw coordinates.
"""
import json
import time
from dataclasses import dataclass
from typing import Optional

import requests

from config import settings


@dataclass
class LocationInfo:
    lat: float
    lon: float
    address: str
    region: str  # coarse admin area, used to match config/local_emergency_directory.json


_cache: dict = {}


def load_camera(camera_id: str) -> dict:
    """Look up a camera's coordinates from config/cameras.json, falling back
    to the single-camera values in .env."""
    try:
        with open(settings.CAMERAS_CONFIG_PATH) as f:
            data = json.load(f)
        for cam in data.get("cameras", []):
            if cam["camera_id"] == camera_id:
                return cam
    except FileNotFoundError:
        pass
    return {
        "camera_id": settings.CAMERA_ID,
        "lat": settings.CAMERA_LAT,
        "lon": settings.CAMERA_LON,
    }


def reverse_geocode(lat: float, lon: float, retries: int = 2, timeout: int = 5) -> LocationInfo:
    """
    Reverse-geocodes lat/lon into a readable address using Nominatim.
    Results are cached in-process since a camera's coordinates never change.
    """
    cache_key = (round(lat, 5), round(lon, 5))
    if cache_key in _cache:
        return _cache[cache_key]

    headers = {"User-Agent": "HighwayGuardianAI/1.0 (emergency-dispatch-system)"}
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 16, "addressdetails": 1}

    for attempt in range(retries + 1):
        try:
            resp = requests.get(settings.NOMINATIM_REVERSE_URL, params=params,
                                 headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            address = data.get("display_name", f"{lat},{lon}")
            addr_parts = data.get("address", {})
            region = (
                addr_parts.get("county")
                or addr_parts.get("state_district")
                or addr_parts.get("city")
                or addr_parts.get("state")
                or "Unknown"
            )
            info = LocationInfo(lat=lat, lon=lon, address=address, region=region)
            _cache[cache_key] = info
            return info
        except requests.RequestException:
            if attempt < retries:
                time.sleep(1)
                continue
            # Fail gracefully -- dispatch must not be blocked by a geocoding outage
            return LocationInfo(lat=lat, lon=lon, address=f"Lat {lat}, Lon {lon} (geocoding unavailable)",
                                 region="Unknown")
