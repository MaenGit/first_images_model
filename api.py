from tensorflow.keras.models import load_model # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

from fastapi import FastAPI, File, UploadFile, Header, HTTPException

import io
# ===== CONFIG =====
API_KEY = "my-secret-one"
size = (224, 224)
work_dir = "./"
np.set_printoptions(suppress=True) # Disable scientific notation for clarity
# Load the model
model = load_model(work_dir+"keras_model.h5", compile=False)
# Load the labels
class_names = open(work_dir+"labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

app = FastAPI() 

@app.post("/predict")
async def predect(
    file: UploadFile = File(...),
    x_api_key : str = Header(None)
):
    #API key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401,detail="Invalid API key")
    
    #Read Image 
    image_bytes = await file.read()
    # Replace this with the path to your image
    immage = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # resizing the image to be at least 224x224 and then cropping from the center

    image = ImageOps.fit(immage, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = np.expand_dims(normalized_image_array,axis=0)

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

    return {
        "label":class_name[2:],
        "confidence":float(confidence_score)
    }









