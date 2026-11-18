from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

app = FastAPI(title="Fake News Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsRequest(BaseModel):
    text: str = Field(min_length=20, max_length=30000)

def load_artifacts():
    paths = {
        "naive_bayes": MODEL_DIR / "naive_bayes.joblib",
        "logistic_regression": MODEL_DIR / "logistic_regression.joblib",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return None
    return {name: joblib.load(path) for name, path in paths.items()}

@app.get("/")
def root():
    return {"message": "Fake News Detection API", "docs": "/docs"}

@app.get("/api/health")
def health():
    artifacts = load_artifacts()
    return {
        "status": "ok",
        "models_ready": artifacts is not None,
        "models": ["Multinomial Naive Bayes", "Logistic Regression"]
    }

@app.post("/api/predict")
def predict(payload: NewsRequest):
    text = payload.text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Please enter at least 20 characters.")
    artifacts = load_artifacts()
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not trained yet. Run: python train.py"
        )

    results = {}
    for key, pipeline in artifacts.items():
        pred = int(pipeline.predict([text])[0])
        probabilities = pipeline.predict_proba([text])[0]
        classes = list(pipeline.classes_)
        confidence = float(probabilities[classes.index(pred)])
        results[key] = {
            "label": "REAL" if pred == 1 else "FAKE",
            "confidence": round(confidence * 100, 2)
        }

    votes = [v["label"] for v in results.values()]
    final_label = max(set(votes), key=votes.count)
    agreement = votes[0] == votes[1]
    return {
        "prediction": final_label,
        "agreement": agreement,
        "models": results,
        "disclaimer": "This is an ML prediction, not a verified fact-check."
    }
