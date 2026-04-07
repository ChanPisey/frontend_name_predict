"""
FastAPI Application for Khmer Gender Classification
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from typing import List, Optional
import uvicorn

from predictor import GenderPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Global predictor ──────────────────────────────────────────────────────────
predictor: GenderPredictor | None = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Loading model...")
    try:
        predictor = GenderPredictor(
            model_path="optimized_gender_model.pt",
            kcc2idx_path="kcc2idx.json",
            fasttext_path="cc.km.300.vec.gz",
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise
    yield
    predictor = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Khmer Gender Classification API",
    description="Gender classification for Khmer first names using Deep Learning",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.genderpredictionkhmer.site",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="Khmer first name", example="ចន្ទា")


class BatchPredictionRequest(BaseModel):
    names: List[str] = Field(..., min_length=1, max_length=100, description="List of Khmer first names")


class PredictionResponse(BaseModel):
    name: str
    gender: Optional[str]
    confidence: Optional[float]
    probability: Optional[float]
    kccs: Optional[List[str]]
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/api.html")
async def api_page():
    return FileResponse("api.html")


@app.get("/features.html")
async def features_page():
    return FileResponse("features.html")


@app.get("/about.html")
async def about_page():
    return FileResponse("about.html")


@app.get("/app.js")
async def app_js():
    return FileResponse("app.js", media_type="application/javascript")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else "N/A",
    }


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
async def predict(request: Request, body: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.predict(body.name)
        return result
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
@limiter.limit("10/minute")
async def batch_predict(request: Request, body: BatchPredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(body.names) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 names per batch request")

    try:
        results = predictor.batch_predict(body.names)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        logger.error("Batch prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Batch prediction failed")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
