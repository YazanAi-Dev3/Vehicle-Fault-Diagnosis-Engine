import json
import pickle
import threading
from collections import deque

import numpy as np
import tensorflow as tf

from config import (
    MODEL_PATH, SCALER_PATH, LE_PATH, CONFIG_PATH,
    FEATURE_COLS, TIME_STEPS, FAULT_THRESHOLD,
    OIL_PRESSURE_PSI, GAP_THRESHOLD_MS, DTC_ALERT_MAP,
)


#  Per-Vehicle Sliding Window Buffer 

class VehicleBuffer:
    """
    Maintains a fixed-size deque of OBD readings for one vehicle.
    Each new reading is appended; once full, the oldest is automatically
    evicted (deque maxlen behaviour), giving us the sliding window.
    Also enforces the trip-isolation gap rule: if the incoming reading
    arrives more than GAP_THRESHOLD_MS after the last one, the buffer
    is cleared before the new reading is added (new trip detected).
    """

    def __init__(self):
        self._buf: deque = deque(maxlen=TIME_STEPS)

    def push(self, reading) -> None:
        """Add a reading, resetting the buffer on trip boundary detection."""
        if self._buf:
            last_ts  = self._buf[-1].timestamp_ms
            gap      = reading.timestamp_ms - last_ts
            if gap > GAP_THRESHOLD_MS or gap < 0:
                # Gap too large or timestamp went backwards → new trip
                self._buf.clear()
        self._buf.append(reading)

    @property
    def is_ready(self) -> bool:
        return len(self._buf) == TIME_STEPS

    @property
    def count(self) -> int:
        return len(self._buf)

    def get_window(self) -> list:
        """Return a stable snapshot of the current window as a plain list."""
        return list(self._buf)


#  Model Service 

class ModelService:
    """
    Singleton that owns:
      - All inference assets (model, scaler, label encoder)
      - One VehicleBuffer per vehicle_id (in-memory, thread-safe)
    """

    def __init__(self):
        self.model         = None
        self.scaler        = None
        self.label_encoder = None
        self.pipeline_cfg  = None
        self._loaded       = False

        # vehicle_id → VehicleBuffer
        self._buffers: dict[str, VehicleBuffer] = {}
        self._lock = threading.Lock()          # guards _buffers for concurrent requests

    #  Asset Loading 

    def load(self):
        self.model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False          )
        self.model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        with open(LE_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(CONFIG_PATH, "r") as f:
            self.pipeline_cfg = json.load(f)

        self._loaded = True
        print(f"Assets loaded. Model input shape: {self.model.input_shape}")

    #  Buffer Management 

    def _get_buffer(self, vehicle_id: str) -> VehicleBuffer:
        """Return the buffer for a vehicle, creating it on first access."""
        with self._lock:
            if vehicle_id not in self._buffers:
                self._buffers[vehicle_id] = VehicleBuffer()
            return self._buffers[vehicle_id]

    @property
    def active_buffer_count(self) -> int:
        with self._lock:
            return len(self._buffers)

    #  Feature Engineering 

    def _engineer_features(self, window: list) -> np.ndarray:
        """
        Replicate the exact feature engineering from the training notebook.
        Input : list of TIME_STEPS OBDReading objects (oldest → newest)
        Output: (TIME_STEPS, n_features) float32 array
        """
        rows = []
        for r in window:
            row = {
                "ENGINE_RPM":          r.ENGINE_RPM,
                "SPEED":               r.SPEED,
                "ENGINE_COOLANT_TEMP": r.ENGINE_COOLANT_TEMP,
                "Oil_Pressure_PSI":    OIL_PRESSURE_PSI,
                "Ratio_RPM_Speed":     r.ENGINE_RPM / (r.SPEED + 1),
                "Thermal_Stress":      r.ENGINE_RPM * r.ENGINE_COOLANT_TEMP,
                "Is_Idle":             float(r.SPEED < 1),
            }
            rows.append([row[col] for col in FEATURE_COLS])

        return np.array(rows, dtype=np.float32)

    #  Main Entry Point 

    def process_reading(self, vehicle_id: str, reading) -> dict:
        """
        Accept one reading, update the vehicle's sliding window,
        and return either a "collecting" status or a full diagnosis.
        """
        buf = self._get_buffer(vehicle_id)
        buf.push(reading)

        if not buf.is_ready:
            # Window not yet full — inform the app of progress
            return {
                "vehicle_id":         vehicle_id,
                "status":             "collecting",
                "readings_collected": buf.count,
                "diagnosis":          None,
                "confidence":         None,
                "alert_required":     None,
                "alert_type":         None,
            }

        # Window is full — run inference on the current 5-reading window
        window  = buf.get_window()
        X       = self._engineer_features(window)
        X_2d    = X.reshape(-1, len(FEATURE_COLS))
        X_scaled = self.scaler.transform(X_2d).reshape(1, TIME_STEPS, len(FEATURE_COLS))

        prob_fault = float(self.model.predict(X_scaled, verbose=0)[0][0])
        is_fault   = prob_fault >= FAULT_THRESHOLD
        label      = "Fault" if is_fault else "Normal"
        confidence = prob_fault if is_fault else (1.0 - prob_fault)
        alert_type = DTC_ALERT_MAP.get("P0133", "ENGINE_FAULT") if is_fault else None

        return {
            "vehicle_id":         vehicle_id,
            "status":             "ready",
            "readings_collected": TIME_STEPS,
            "diagnosis":          label,
            "confidence":         round(confidence, 4),
            "alert_required":     is_fault,
            "alert_type":         alert_type,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Global singleton — imported by main.py
model_service = ModelService()
