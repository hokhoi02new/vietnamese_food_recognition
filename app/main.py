import uvicorn
import numpy as np
import cv2
import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from config.config import IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES, BACKBONE, SAVED_MODEL_DIR
import gdown

app = FastAPI(title="Vietnamese food recongization API", version="1.0")

UPLOAD_FOLDER = "uploads"
model = None

def load_model(backbone=BACKBONE):
    """Load model đã train theo backbone"""
    global model
    model_path = os.path.join(SAVED_MODEL_DIR, f"{backbone}_model.keras")

    if not os.path.exists(model_path):
        os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
        print("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1PM5CsIEGAeJY2ezkzn3lgqrFq8ZJpese"   
        gdown.download(url, model_path, quiet=False)
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model {backbone} loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def preprocess_image_cv2(image_path: str):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    try:
        file_bytes = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        img_array = preprocess_image_cv2(file_path)

        preds = model.predict(img_array, verbose=0)
        idx = np.argmax(preds[0])
        confidence = float(preds[0][idx])

        return {
            "filename": file.filename,
            "predicted_label": CLASS_NAMES[idx],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/predict-batch")
async def predict_multiple(files: List[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    results = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            file_bytes = await file.read()
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            img_array = preprocess_image_cv2(file_path)

            preds = model.predict(img_array, verbose=0)
            idx = np.argmax(preds[0])
            confidence = float(preds[0][idx])

            results.append({
                "filename": file.filename,
                "predicted_label": CLASS_NAMES[idx],
                "confidence": round(confidence, 4)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return {"predictions": results}

@app.get("/")
async def root():
    return {"message": "Vietnamese food recongization API is running"}

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True, "model_type": BACKBONE, "message": "API is running healthy"}

@app.on_event("startup")
async def startup_event():
    load_model(backbone=BACKBONE)
