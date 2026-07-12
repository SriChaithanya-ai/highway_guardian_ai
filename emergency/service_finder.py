"""
Finds the nearest police station, hospital, and notes that ambulance
dispatch should go through the matched hospital/ambulance entry.

Primary source: OpenStreetMap Overpass API (free, no key) -- finds the
nearest amenity=police / amenity=hospital node and its tagged phone number
if present.

Fallback: config/local_emergency_directory.json, matched by the region
name returned from reverse geocoding, for when OSM has no phone tag
(very common) or no matching node nearby.
"""
import json
import math
from dataclasses import dataclass
from typing import Optional

import requests

from config import settings


@dataclass
class EmergencyContact:
    name: str
    service_type: str  # "police" | "hospital" | "ambulance"
    lat: Optional[float]
    lon: Optional[float]
    phone: Optional[str]
    source: str  # "osm" | "local_directory" | "national_fallback"


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _query_overpass(lat: float, lon: float, amenity: str, radius_m: int) -> Optional[dict]:
    query = f"""
    [out:json][timeout:10];
    node["amenity"="{amenity}"](around:{radius_m},{lat},{lon});
    out body;
    """
    try:
        resp = requests.post(settings.OVERPASS_API_URL, data={"data": query}, timeout=12)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except requests.RequestException:
        return None

    if not elements:
        return None

    best, best_dist = None, float("inf")
    for el in elements:
        d = _haversine_m(lat, lon, el["lat"], el["lon"])
        if d < best_dist:
            best, best_dist = el, d
    return best


def _load_local_directory() -> dict:
    try:
        with open(settings.POLICE_DIRECTORY_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"stations": [], "default_national_emergency_number": "112"}


def _nearest_from_local_directory(lat: float, lon: float, service_type: str,
                                   region: str) -> Optional[EmergencyContact]:
    directory = _load_local_directory()
    candidates = [s for s in directory.get("stations", []) if s["type"] == service_type]
    if not candidates:
        return None

    # Prefer same-region matches, fall back to nearest by distance
    region_matches = [s for s in candidates if s.get("region", "").lower() == region.lower()]
    pool = region_matches if region_matches else candidates

    best, best_dist = None, float("inf")
    for s in pool:
        d = _haversine_m(lat, lon, s["lat"], s["lon"])
        if d < best_dist:
            best, best_dist = s, d

    if best is None:
        return None
    return EmergencyContact(
        name=best["name"], service_type=service_type,
        lat=best["lat"], lon=best["lon"], phone=best["phone"],
        source="local_directory",
    )


def find_nearest(lat: float, lon: float, service_type: str, region: str) -> EmergencyContact:
    """
    service_type: "police" | "hospital"
    Tries OSM first (live, up-to-date locations), falls back to the local
    directory for a verified phone number, and finally to the national
    emergency number so dispatch is never silently skipped.
    """
    osm_amenity = {"police": "police", "hospital": "hospital"}[service_type]
    osm_node = _query_overpass(lat, lon, osm_amenity, settings.NEAREST_SERVICE_RADIUS_M)

    if osm_node:
        tags = osm_node.get("tags", {})
        phone = tags.get("phone") or tags.get("contact:phone")
        name = tags.get("name", f"Nearest {service_type}")
        if phone:
            return EmergencyContact(
                name=name, service_type=service_type,
                lat=osm_node["lat"], lon=osm_node["lon"], phone=phone, source="osm",
            )
        # OSM found the nearest station but no phone tag -- try local directory
        local = _nearest_from_local_directory(lat, lon, service_type, region)
        if local:
            local.name = f"{name} (contact via directory: {local.name})"
            return local

    # No OSM node in range at all -- try local directory directly
    local = _nearest_from_local_directory(lat, lon, service_type, region)
    if local:
        return local

    # Last resort: national emergency number, so a human dispatcher can route it
    directory = _load_local_directory()
    national_number = directory.get("default_national_emergency_number", "112")
    return EmergencyContact(
        name=f"National Emergency Line ({service_type})", service_type=service_type,
        lat=None, lon=None, phone=national_number, source="national_fallback",
    )
