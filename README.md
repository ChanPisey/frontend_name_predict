# Khmer Gender Classification API

A FastAPI-based REST API for predicting gender from Khmer first names using Khmer Character Cluster (KCC) segmentation, FastText embeddings, and BiLSTM neural network.

## Architecture

- **KCC (Khmer Character Cluster)**: Segments Khmer text into meaningful character clusters
- **FastText**: Pre-trained word embeddings for Khmer language
- **BiLSTM**: Bidirectional LSTM neural network for sequence classification

## Project Structure

```
.
├── main.py                 # FastAPI application
├── model.py               # BiLSTM model architecture
├── predictor.py          # Prediction service
├── utils.py              # KCC helper functions
├── requirements.txt      # Python dependencies
├── optimized_gender_model.pt  # Trained PyTorch model
├── kcc2idx.json         # KCC vocabulary mapping
└── cc.km.300.vec.gz     # FastText embeddings
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following files in the project directory:
   - `optimized_gender_model.pt` (trained model)
   - `kcc2idx.json` (vocabulary)
   - `cc.km.300.vec.gz` (FastText embeddings)

## Running the API

Start the server:
```bash
python main.py
```

Or use uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

## API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "name": "ចន្ទា"
}
```

Response:
```json
{
  "name": "ចន្ទា",
  "gender": "Female",
  "confidence": 99.91,
  "probability": 0.9991,
  "kccs": ["ច", "ន្ទា"]
}
```

### 3. Batch Prediction
```bash
POST /batch_predict
Content-Type: application/json

{
  "names": ["ចន្ទា", "សុខ", "រដ្ឋា"]
}
```

Response:
```json
{
  "predictions": [
    {
      "name": "ចន្ទា",
      "gender": "Female",
      "confidence": 99.91,
      "probability": 0.9991,
      "kccs": ["ច", "ន្ទា"]
    },
    {
      "name": "សុខ",
      "gender": "Male",
      "confidence": 92.36,
      "probability": 0.0764,
      "kccs": ["សុ", "ខ"]
    }
  ],
  "count": 2
}
```

## Interactive Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Example Usage with cURL

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"name": "ចន្ទា"}'

# Batch prediction
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"names": ["ចន្ទា", "សុខ", "រដ្ឋា"]}'
```

## Example Usage with Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"name": "ចន្ទា"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/batch_predict",
    json={"names": ["ចន្ទា", "សុខ", "រដ្ឋា"]}
)
print(response.json())
```

## Model Details

- **Embedding Dimension**: 300
- **Hidden Dimension**: 128
- **Number of LSTM Layers**: 5
- **Dropout**: 0.3
- **Max Sequence Length**: 30
- **Vocabulary Size**: 1915 KCCs

## Gender Classification

- **0 = Male**
- **1 = Female**
- Threshold: 0.5 (probability > 0.5 = Female, else Male)

## Notes

- The model expects Khmer first names as input
- Names are automatically cleaned and segmented into KCCs
- The API supports CORS for cross-origin requests
- Maximum batch size: 100 names per request
"# frontend_name_predict" 
