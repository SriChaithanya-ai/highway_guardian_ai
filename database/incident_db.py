"""
Incident logging database (SQLite via SQLAlchemy Core -- no external DB
server required, works out of the box for a single-site deployment).
"""
import datetime as dt
import json

from sqlalchemy import (Column, DateTime, Float, Integer, String, Text,
                         create_engine, insert, select)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import settings

Base = declarative_base()


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
    camera_id = Column(String, index=True)
    lat = Column(Float)
    lon = Column(Float)
    address = Column(Text)
    severity = Column(String)
    severity_confidence = Column(Float)
    classifier_confidence = Column(Float)
    vehicle_count = Column(Integer)
    vehicle_classes = Column(Text)     # JSON list, e.g. ["car", "truck"]
    snapshot_path = Column(Text)
    police_contact = Column(Text)      # JSON: {name, phone, source}
    hospital_contact = Column(Text)    # JSON: {name, phone, source}
    dispatch_log = Column(Text)        # JSON list of dispatch records
    status = Column(String, default="dispatched")  # dispatched | false_alarm | resolved


_engine = create_engine(f"sqlite:///{settings.DB_PATH}", future=True)
Base.metadata.create_all(_engine)
Session = sessionmaker(bind=_engine, future=True)


def log_incident(camera_id: str, lat: float, lon: float, address: str,
                  severity: str, severity_confidence: float, classifier_confidence: float,
                  vehicle_count: int, vehicle_classes: list, snapshot_path: str,
                  police_contact: dict, hospital_contact: dict, dispatch_log: list) -> int:
    with Session() as session:
        incident = Incident(
            camera_id=camera_id, lat=lat, lon=lon, address=address,
            severity=severity, severity_confidence=severity_confidence,
            classifier_confidence=classifier_confidence,
            vehicle_count=vehicle_count, vehicle_classes=json.dumps(vehicle_classes),
            snapshot_path=snapshot_path,
            police_contact=json.dumps(police_contact),
            hospital_contact=json.dumps(hospital_contact),
            dispatch_log=json.dumps(dispatch_log),
        )
        session.add(incident)
        session.commit()
        return incident.id


def get_recent_incidents(limit: int = 50):
    with Session() as session:
        stmt = select(Incident).order_by(Incident.timestamp.desc()).limit(limit)
        return list(session.scalars(stmt))


def get_incident_count_since(camera_id: str, since: dt.datetime) -> int:
    with Session() as session:
        stmt = select(Incident).where(Incident.camera_id == camera_id, Incident.timestamp >= since)
        return len(list(session.scalars(stmt)))
