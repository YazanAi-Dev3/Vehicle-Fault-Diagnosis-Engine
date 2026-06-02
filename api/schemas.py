from pydantic import BaseModel, field_validator
from typing import Optional
from config import GAP_THRESHOLD_MS


class OBDReading(BaseModel):
    """Single OBD-II sensor snapshot sent by the Android app every minute."""
    ENGINE_RPM:          float
    SPEED:               float
    ENGINE_COOLANT_TEMP: float
    timestamp_ms:        int       # Unix epoch milliseconds


class PredictRequest(BaseModel):
    """
    One reading per request.
    The API maintains the sliding window buffer internally per vehicle.
    The Android app only needs to send this payload every ~60 seconds.
    """
    vehicle_id: str
    reading:    OBDReading         # single reading, not a list

    @field_validator("vehicle_id")
    @classmethod
    def validate_vehicle_id(cls, v):
        if not v or not v.strip():
            raise ValueError("vehicle_id must not be empty.")
        return v.strip()


class PredictResponse(BaseModel):
    """
    Response returned after every single reading.

    status:
      - "collecting"  → buffer not yet full, no diagnosis available
      - "ready"       → buffer full, diagnosis is valid

    When status == "collecting":
      - diagnosis, confidence, alert_required, alert_type are all None
      - readings_collected tells the app how many readings have arrived so far

    When status == "ready":
      - all fields are populated with a real inference result
    """
    vehicle_id:         str
    status:             str                  # "collecting" | "ready"
    readings_collected: int                  # 1–5, helps app show progress
    diagnosis:          Optional[str]        # "Fault" | "Normal" | None
    confidence:         Optional[float]      # 0–1 | None
    alert_required:     Optional[bool]
    alert_type:         Optional[str]
    timestamp:          str                  # ISO-8601 UTC


class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    active_buffers:  int                     # how many vehicles are currently tracked
