import os

# Force CPU-only mode by disabling CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Prevent TensorFlow from using GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow import lite
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import uvicorn
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Model and class configurations
MODEL_PATH = "model.tflite"  # TensorFlow Lite model path
KERAS_MODEL_PATH = "adam_cnn.keras"  # Original Keras model path
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
        # Ensure that the image is loaded in the correct format with Pillow (PIL)
        img = load_img(img_path, target_size=target_size)  # Use load_img from Keras
        img_array = img_to_array(img)  # Convert image to array
        img_array = img_array / 255.0  # Normalize the image to the range [0, 1]
        return np.expand_dims(img_array, axis=0)  # Add batch dimension for prediction
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

# Retrain the model
@app.post("/retrain/")
async def retrain_model(dataset_dir: str):
    try:
        # Check dataset directory exists
        if not os.path.exists(dataset_dir):
            raise HTTPException(status_code=404, detail="Dataset directory not found.")

        # Load the original Keras model
        if not os.path.exists(KERAS_MODEL_PATH):
            raise HTTPException(status_code=404, detail="Original Keras model not found.")
        model = load_model(KERAS_MODEL_PATH)

        # Check the model's input shape and ensure it aligns with the dataset images
        target_size = (150, 150)  # Adjust to the target size for your images
        print(f"Model input shape: {model.input_shape}, target size: {target_size}")

        # Prepare data generators
        datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(dataset_dir, target_size=target_size, batch_size=32, subset='training')
        val_gen = datagen.flow_from_directory(dataset_dir, target_size=target_size, batch_size=32, subset='validation')

        # Retrain the model
        model.fit(train_gen, validation_data=val_gen, epochs=5)

        # Save retrained Keras model
        model.save(KERAS_MODEL_PATH)

        # Convert retrained model to TensorFlow Lite
        from tensorflow.lite import TFLiteConverter
        converter = TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the updated TensorFlow Lite model
        with open(MODEL_PATH, "wb") as f:
            f.write(tflite_model)

        return {"message": "Model retrained and converted to TensorFlow Lite successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

# Health check
@app.get("/")
def root():
    return {"message": "FastAPI server is running."}

# Load the TensorFlow Lite model on server start
load_tflite_model()