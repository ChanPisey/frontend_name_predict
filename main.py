"""
FastAPI Application for Khmer Gender Classification
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from predictor import GenderPredictor
# Initialize FastAPI app
app = FastAPI(
    title="Khmer Gender Classification API",
    description="Gender classification for Khmer first names using Deep Learning",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Global predictor instance
predictor = None


# Request/Response Models
class PredictionRequest(BaseModel):
    name: str = Field(..., description="Khmer first name", example="ចន្ទា", max_length=100)


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


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the main frontend page"""
    return FileResponse("index.html")


@app.get("/about", response_class=FileResponse)
async def about():
    return FileResponse("about.html")


@app.get("/features", response_class=FileResponse)
async def features():
    return FileResponse("features.html")


@app.get("/api-page", response_class=FileResponse)
async def api_page():
    return FileResponse("api.html")


@app.get("/app.js")
async def serve_app_js():
    return FileResponse("app.js", media_type="application/javascript")


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
        reload=False
    )
