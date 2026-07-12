"""
Emergency Notification module.

This is the "no human in the loop" piece: once an incident is CONFIRMED by
the verification module, this fires automatically -- no dashboard operator
needs to click anything.

Channels:
  - Automated voice call (Twilio) that reads out an incident summary via
    text-to-speech (TwiML <Say>) to the nearest police station AND the
    nearest hospital/ambulance line.
  - SMS to the same numbers with the same summary + Google Maps link.
  - Email to a configured hospital/control-room address as a durable record.

If Twilio/SMTP credentials are not configured (e.g. during local testing),
calls are logged instead of sent, so the pipeline never crashes because a
secret is missing -- but nothing is silently "faked" as delivered.
"""
import logging
import smtplib
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import List, Optional

from config import settings

logger = logging.getLogger("emergency.notifier")


@dataclass
class DispatchRecord:
    channel: str          # "call" | "sms" | "email"
    recipient_name: str
    recipient_contact: str
    status: str            # "sent" | "failed" | "skipped_no_credentials"
    detail: str = ""


def _twilio_client():
    if not (settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN and settings.TWILIO_FROM_NUMBER):
        return None
    from twilio.rest import Client
    return Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)


def build_incident_message(incident: dict) -> str:
    return (
        f"Highway Guardian AI Alert. A {incident['severity']} traffic accident was detected "
        f"on camera {incident['camera_id']} near {incident['address']}. "
        f"{incident['vehicle_count']} vehicles appear involved. "
        f"Coordinates: {incident['lat']}, {incident['lon']}. "
        f"Map link: https://maps.google.com/?q={incident['lat']},{incident['lon']}. "
        f"This is an automated alert. Please dispatch emergency response immediately."
    )


def place_automated_call(to_number: str, recipient_name: str, message: str) -> DispatchRecord:
    client = _twilio_client()
    if client is None:
        logger.warning("Twilio not configured -- call to %s (%s) skipped.", recipient_name, to_number)
        return DispatchRecord("call", recipient_name, to_number, "skipped_no_credentials")

    twiml = f"<Response><Say voice='alice' loop='2'>{message}</Say></Response>"
    try:
        call = client.calls.create(twiml=twiml, to=to_number, from_=settings.TWILIO_FROM_NUMBER)
        return DispatchRecord("call", recipient_name, to_number, "sent", detail=f"sid={call.sid}")
    except Exception as e:  # Twilio raises various exception types
        logger.exception("Failed to place call to %s", to_number)
        return DispatchRecord("call", recipient_name, to_number, "failed", detail=str(e))


def send_sms(to_number: str, recipient_name: str, message: str) -> DispatchRecord:
    client = _twilio_client()
    if client is None:
        logger.warning("Twilio not configured -- SMS to %s (%s) skipped.", recipient_name, to_number)
        return DispatchRecord("sms", recipient_name, to_number, "skipped_no_credentials")

    try:
        sms = client.messages.create(body=message, to=to_number, from_=settings.TWILIO_FROM_NUMBER)
        return DispatchRecord("sms", recipient_name, to_number, "sent", detail=f"sid={sms.sid}")
    except Exception as e:
        logger.exception("Failed to send SMS to %s", to_number)
        return DispatchRecord("sms", recipient_name, to_number, "failed", detail=str(e))


def send_email(to_email: str, recipient_name: str, subject: str, message: str) -> DispatchRecord:
    if not (settings.SMTP_HOST and settings.SMTP_USERNAME and settings.SMTP_PASSWORD):
        logger.warning("SMTP not configured -- email to %s skipped.", to_email)
        return DispatchRecord("email", recipient_name, to_email, "skipped_no_credentials")

    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_FROM_EMAIL
    msg["To"] = to_email

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM_EMAIL, [to_email], msg.as_string())
        return DispatchRecord("email", recipient_name, to_email, "sent")
    except Exception as e:
        logger.exception("Failed to email %s", to_email)
        return DispatchRecord("email", recipient_name, to_email, "failed", detail=str(e))


def dispatch_all(incident: dict, police_contact, hospital_contact,
                  control_room_email: Optional[str] = None) -> List[DispatchRecord]:
    """
    Fires all channels automatically, no confirmation step. Called
    immediately once the pipeline confirms an incident.
    """
    message = build_incident_message(incident)
    records: List[DispatchRecord] = []

    if police_contact and police_contact.phone:
        records.append(place_automated_call(police_contact.phone, police_contact.name, message))
        records.append(send_sms(police_contact.phone, police_contact.name, message))

    if hospital_contact and hospital_contact.phone:
        records.append(place_automated_call(hospital_contact.phone, hospital_contact.name, message))
        records.append(send_sms(hospital_contact.phone, hospital_contact.name, message))

    if control_room_email:
        records.append(send_email(control_room_email, "Control Room",
                                   subject=f"[URGENT] Accident detected - {incident['camera_id']}",
                                   message=message))

    for rec in records:
        logger.info("Dispatch: %s -> %s (%s) [%s] %s",
                    rec.channel, rec.recipient_name, rec.recipient_contact, rec.status, rec.detail)

    return records
