from fastapi import FastAPI, UploadFile, File
from ml.bitalino_model import train_model_from_folders, score_file
import shutil
import os

app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = train_model_from_folders("ml/CALM/*.txt")
        print("✅ Model trained successfully.")
    except FileNotFoundError:
        print("⚠️ No baseline files found. Model not trained yet.")
        model = None


@app.get("/")
def root():
    return {"message": "BITalino Stress API running 🚀"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not trained yet. No baseline files found."}

    temp_path = f"ml/temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    avg_score, scores, preds = score_file(model, temp_path)
    os.remove(temp_path)

    return {
        "average_score": avg_score,
        "total_windows": len(scores),
        "anomalies": int((preds == -1).sum())
    }
