import os

# Force CPU-only mode by disabling CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Prevent TensorFlow from using GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import lite
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Model and class configurations
MODEL_PATH = "model.tflite"  # Use TensorFlow Lite model
CLASS_LABELS = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
RECYCLABLE_CLASSES = {'cardboard', 'glass', 'metal', 'paper', 'plastic'}
interpreter = None
input_details = None
output_details = None

# Load TensorFlow Lite model
def load_tflite_model():
    global interpreter, input_details, output_details
    try:
        interpreter = lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded successfully.")
    except Exception as e:
        print(f"Error loading TensorFlow Lite model: {e}")

# Preprocess uploaded image
def preprocess_image(img_path, target_size):
    try:
        img = load_img(img_path, target_size=target_size)  # Use load_img
        img_array = img_to_array(img)  # Use img_to_array
        img_array = img_array / 255.0  # Normalize
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")

# Predict the class of an image
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if interpreter is None:
        return JSONResponse(status_code=500, content={"message": "Model is not loaded."})

    try:
        # Save uploaded file locally
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Preprocess the image
        target_size = input_details[0]['shape'][1:3]  # Height and width from model input shape
        img_array = preprocess_image(temp_file_path, target_size)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(input_details[0]['dtype']))

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(predictions)
        predicted_label = CLASS_LABELS[predicted_class_index]
        category = "Recyclable" if predicted_label in RECYCLABLE_CLASSES else "Non-Recyclable"

        # Clean up
        os.remove(temp_file_path)

        return {"predicted_label": predicted_label, "category": category}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Health check
@app.get("/")
def root():
    return {"message": "FastAPI server is running."}

# Load the TensorFlow Lite model on server start
load_tflite_model()
