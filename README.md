# Intelligent Vehicle Fault Diagnosis Engine

An end-to-end MLOps pipeline and backend architecture for vehicle diagnostics. This system processes time-series OBD-II sensor data to classify vehicle states as normal or faulty, providing real-time predictive maintenance capabilities.

## Architecture Evolution

This repository captures the transition from a traditional machine learning approach to an advanced deep learning architecture specifically tailored for temporal sensor data.

1. **Phase 1: Baseline Random Forest Cascade (`notebooks/RF_Baseline_Cascade.ipynb`)**
   - The initial approach utilized a cascade of Random Forest classifiers. A "Gatekeeper" binary classifier first determined if the vehicle was in a normal or faulty state. If faulty, a "Specialist" multi-class model identified the specific OBD-II error code. 
   - While effective for tabular snapshots, this approach struggled to capture the sequential degradation of engine components over time.

2. **Phase 2: Temporal LSTM Engine (`notebooks/LSTM_Comp.ipynb`)**
   - To better leverage the time-series nature of OBD-II data, we transitioned to a stateful LSTM (Long Short-Term Memory) network.
   - The LSTM effectively models temporal patterns across sequential readings, outperforming the baseline Random Forest by capturing the trajectory of thermal stress and RPM/Speed ratios across entire trips.
   - **This LSTM architecture is the final production model powering the backend.**

## End-to-End Operational Pipeline

The `api/` directory contains the production-ready FastAPI backend designed to serve the LSTM model in a highly concurrent environment.

### Stateful Sliding Window Inference
The `model_service.py` implements a robust in-memory buffer system that handles asynchronous streams of sensor data:
- **Vehicle Tracking**: The service tracks independent reading buffers for unique `vehicle_id`s.
- **Trip Boundary Detection**: If a reading is delayed by more than ~60 seconds, the system detects a "new trip" and automatically resets the sequence buffer.
- **Sliding Window**: The system requires a window of 5 consecutive readings to form a complete temporal sequence. Once 5 readings are collected, a diagnosis is yielded. Each subsequent reading slides the window forward by one step, providing continuous real-time monitoring.

## Tech Stack
- **Deep Learning**: TensorFlow / Keras (LSTM)
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Backend API**: FastAPI, Pydantic, Uvicorn
- **Environment**: Python 3.10+

## API Usage

The system exposes a unified diagnosis endpoint designed for a connected Android application or edge device.

**POST `/api/v1/predict`**

```json
{
  "vehicle_id": "VIN-ABC123",
  "reading": {
    "ENGINE_RPM": 1200.0,
    "SPEED": 40.0,
    "ENGINE_COOLANT_TEMP": 88.0,
    "timestamp_ms": 1700000060000
  }
}
```

The system manages the buffer internally. It returns a `collecting` status until the window is full, after which it returns a `ready` status with the diagnosis.

## ⚠️ Note on Weights and Datasets

**Heavy artifacts (such as `.h5`, `.keras`, and `.joblib` model weights) and raw CSV/JSON datasets are intentionally excluded from this repository to maintain a clean and lightweight version control history.**

To run the backend API locally:
1. Run the `notebooks/LSTM_Comp.ipynb` notebook to train the network on your local machine using your own OBD-II datasets.
2. Ensure the resulting `.keras` / `.h5` weights and `.pkl` encoders are saved in the `deployment_assets/` directory.
3. Install dependencies: `pip install -r requirements.txt`
4. Start the FastAPI server: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

## License
MIT License
