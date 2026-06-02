from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import PredictRequest, PredictResponse, HealthResponse
from model_service import model_service


#  Lifespan: load assets once at startup 

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


#  App Initialization 

app = FastAPI(
    title="Intelligent Car Fault Diagnosis API",
    description=(
        "LSTM-based fault detection using a stateful sliding window. "
        "Send one OBD-II reading per request every ~60 seconds. "
        "The API buffers readings per vehicle and returns a diagnosis "
        "once 5 readings have been collected. Each subsequent reading "
        "slides the window forward by one step."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


#  Routes 

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Liveness probe.
    active_buffers shows how many vehicles are currently being tracked.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        active_buffers=model_service.active_buffer_count,
    )


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Diagnosis"])
def predict(request: PredictRequest):
    """
    Sliding-window diagnosis endpoint.

    ## Android app contract
    - Send **one reading** every ~60 seconds.
    - Include `timestamp_ms` (Unix epoch in milliseconds) in every reading.
    - No need to manage any local buffer — the API handles everything.

    ## Response lifecycle
    | readings received | status       | diagnosis |
    |-------------------|--------------|-----------|
    | 1                 | collecting   | null      |
    | 2                 | collecting   | null      |
    | 3                 | collecting   | null      |
    | 4                 | collecting   | null      |
    | 5+                | ready        | Fault / Normal |

    ## Trip boundary handling
    If the gap between two consecutive readings exceeds 60 seconds,
    the buffer is automatically reset (new trip detected).
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    try:
        result = model_service.process_reading(request.vehicle_id, request.reading)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return PredictResponse(
        **result,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
