"""
FastAPI Application for Khmer Gender Classification
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from predictor import GenderPredictor
from fastapi import FastAPI
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# Initialize FastAPI app
app = FastAPI(
    title="Khmer Gender Classification API",
    description="Gender classification for Khmer first names using KCC + FastText + BiLSTM",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global predictor instance
predictor = None


# Request/Response Models
class PredictionRequest(BaseModel):
    name: str = Field(..., description="Khmer first name", example="ចន្ទា")


class BatchPredictionRequest(BaseModel):
    names: List[str] = Field(..., description="List of Khmer first names", example=["ចន្ទា", "សុខ"])


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


@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global predictor
    print("Loading model...")
    try:
        predictor = GenderPredictor(
            model_path="optimized_gender_model.pt",
            kcc2idx_path="kcc2idx.json",
            fasttext_path="cc.km.300.vec.gz"
        )
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Khmer Gender Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else "N/A"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict gender for a single Khmer first name

    Args:
        request: PredictionRequest containing the name

    Returns:
        PredictionResponse with gender prediction
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.name or not request.name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    try:
        result = predictor.predict(request.name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict gender for multiple Khmer first names

    Args:
        request: BatchPredictionRequest containing list of names

    Returns:
        BatchPredictionResponse with predictions for all names
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.names:
        raise HTTPException(status_code=400, detail="Names list cannot be empty")

    if len(request.names) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 names per batch request")

    try:
        results = predictor.batch_predict(request.names)
        return {
            "predictions": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    # Run the app
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
