import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io

# Create the FastAPI app
app = FastAPI()

# Load the transfer learning model from the local directory
model = tf.keras.models.load_model('models/adam_cnn.keras')  # Adjust the path as necessary
class_names = ['non-recyclable', 'recyclable']

def read_imagefile(file) -> Image.Image:
    """Read image from the uploaded file."""
    image = Image.open(io.BytesIO(file))
    return image

def preprocess_image(image: Image.Image):
    """Preprocess the image for the model."""
    image = image.resize((224, 224))  # Resize to the model's input shape
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict the class of the uploaded image."""
    image = read_imagefile(await file.read())
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    return {"class": predicted_class, "confidence": float(np.max(predictions))}

# To run the FastAPI app, use the command:
# uvicorn app:app --host 0.0.0.0 --port 8000
