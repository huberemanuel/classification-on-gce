import base64
import json
from io import BytesIO

import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, Depends, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from classifier.train import load_model

app = FastAPI()
app.mount("/static", StaticFiles(directory="classifier/web/static"), name="static")
templates = Jinja2Templates(directory="classifier/web/templates")

def preprocess_img(img: bytes):
    input_img = BytesIO(base64.urlsafe_b64decode(img))
    # Reading image and resing to nn input
    input_img = Image.open(input_img).resize((32, 32)
    # Converting to np.array and removing alpha channel
    input_img = np.array(input_img))[:,:,:3]
    return tf.expand_dims(input_img, axis=0)

async def load_saved_model():
    # TODO: Load trained model from GCP bucket
    return load_model()

@app.post("/classify")
async def root(img: bytes = File(...), model: tf.keras.Model = Depends(load_saved_model)):
    result = {"prediction" :"Empty", "probability" :{}}

    input_img = preprocess_img(img)
    res = model.predict(input_img)[0]
    print(res, len(res))
    
    result["prediction"] = str(np.argmax(res))
    result["probability"] = res[np.argmax(res)] * 100

    return result

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})