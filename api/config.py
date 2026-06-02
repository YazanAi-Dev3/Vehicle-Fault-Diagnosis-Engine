import os

#  Asset Paths 
# Resolve relative to this file so the API works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "..", "deployment_assets")

MODEL_PATH = os.path.join(ASSETS_DIR, "lstm_gatekeeper.h5")
SCALER_PATH  = os.path.join(ASSETS_DIR, "scaler.pkl")
LE_PATH      = os.path.join(ASSETS_DIR, "label_encoder.pkl")
CONFIG_PATH  = os.path.join(ASSETS_DIR, "pipeline_config.json")

#  Pipeline Constants (mirrors training notebook) 
TIME_STEPS        = 5
GAP_THRESHOLD_MS  = 60_000       # max allowed gap between consecutive readings
FAULT_THRESHOLD   = 0.5          # sigmoid decision boundary
OIL_PRESSURE_PSI  = 30.0         # constant feature (no OBD-II sensor in dataset)

FEATURE_COLS = [
    "ENGINE_RPM",
    "SPEED",
    "ENGINE_COOLANT_TEMP",
    "Oil_Pressure_PSI",
    "Ratio_RPM_Speed",
    "Thermal_Stress",
    "Is_Idle",
]

#  Alert Mapping 
# Maps DTC codes the model was trained on to human-readable alert types
DTC_ALERT_MAP = {
    "P0133": "O2_SENSOR_SLOW_RESPONSE",
    "C0300": "WHEEL_SPEED_SENSOR_FAULT",
}
