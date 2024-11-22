import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
import uvicorn
from typing import Dict, List

# Initialize FastAPI app
app = FastAPI()

# Model and class configurations
MODEL_PATH = "adam_cnn.keras" 
CLASS_LABELS = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
RECYCLABLE_CLASSES = {'cardboard', 'glass', 'metal', 'paper', 'plastic'}
model = None

# Load model
def load_model_file():
    global model
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

# Preprocess uploaded image
def preprocess_image(img_path, target_size):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# Predict the class of an image
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"message": "Model is not loaded."})

    try:
        # Save uploaded file locally
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Preprocess the image
        target_size = model.input_shape[1:3]
        img_array = preprocess_image(temp_file_path, target_size)

        # Predict
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_label = CLASS_LABELS[predicted_class_index]
        category = "Recyclable" if predicted_label in RECYCLABLE_CLASSES else "Non-Recyclable"

        # Clean up
        os.remove(temp_file_path)

        return {"predicted_label": predicted_label, "category": category}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Retrain the model
@app.post("/retrain/")
async def retrain_model(dataset_dir: str):
    try:
        if not os.path.exists(dataset_dir):
            raise HTTPException(status_code=404, detail="Dataset directory not found.")

        # Prepare data generators
        datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(dataset_dir, target_size=(150, 150), batch_size=32, subset='training')
        val_gen = datagen.flow_from_directory(dataset_dir, target_size=(150, 150), batch_size=32, subset='validation')

        # Retrain model
        global model
        model.fit(train_gen, validation_data=val_gen, epochs=5)

        # Save retrained model
        model.save(MODEL_PATH)
        return {"message": "Model retrained and saved successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Health check
@app.get("/")
def root():
    return {"message": "FastAPI server is running."}

# Load the model on server start
load_model_file()
