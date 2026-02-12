from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ml.bitalino_model import train_model_from_folders, score_file
import shutil
import os

app = FastAPI()

# Allow requests from your frontend (Next.js dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your Next.js dev URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Train model once at startup
model = train_model_from_folders("ml/CALM/*.txt")

# Folder to temporarily save uploaded files
UPLOAD_DIR = "ml/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def classify_stress(avg_score, preds):
    # Proportion of windows flagged as stressed
    n_windows = len(preds)
    n_anom = (preds == -1).sum()
    stress_ratio = n_anom / n_windows

    # Decide stress level
    if avg_score < 0 and stress_ratio < 0.2:
        level = "calm"
        message = "You look calm and relaxed. Keep it up!"
    elif stress_ratio < 0.5:
        level = "moderate"
        message = "You seem okay, but there are some signs of stress."
    else:
        level = "stressed"
        message = "You look more stressed than normal. Take a deep breath!"

    return level, message


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Score the file
    avg_score, anomaly_score, preds = score_file(model, file_path)

    level, message = classify_stress(avg_score, preds)

    # Optionally delete the uploaded file
    os.remove(file_path)

    return {
        "level": level,
        "message": message,  # convert numpy array to list
    }
