import base64
import json
from io import BytesIO
import os
import shutil

import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, Depends, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google.cloud import storage


def download_model():
    bucket_name = "gcp_iac_bucket_dev"
    trained_models = tf.io.gfile.listdir(f"gs://{bucket_name}/models")
    best_model = sorted(trained_models)[-1]
    
    temp_location = "digits_model_best"
    if os.path.exists(temp_location):
        shutil.rmtree(temp_location)

    os.mkdir(temp_location)
    for root, dirs, files in tf.io.gfile.walk(f"gs://{bucket_name}/models/{best_model}"):
        for file_name in files:
            out_path = os.path.join(temp_location, root.split(best_model)[-1])
            temp_file = open(os.path.join(out_path, file_name), "wb")
            remote_file = tf.io.gfile.GFile(os.path.join(root, file_name), mode="rb")
            temp_file.write(remote_file.read())
            remote_file.close()
            temp_file.close()
        for dir_name in dirs:
            out_dir = os.path.join(temp_location, dir_name)
            os.makedirs(out_dir)
            for _, _, subfiles in tf.io.gfile.walk(os.path.join(root, dir_name)):
                for subfile_name in subfiles:
                    temp_file = open(os.path.join(out_dir, subfile_name), "wb")
                    remote_file = tf.io.gfile.GFile(os.path.join(root, dir_name, subfile_name), mode="rb")
                    temp_file.write(remote_file.read())
                    remote_file.close()
                    temp_file.close()

app = FastAPI()
app.mount("/static", StaticFiles(directory="classifier/web/static"), name="static")
templates = Jinja2Templates(directory="classifier/web/templates")
download_model()

def preprocess_img(img: bytes):
    input_img = BytesIO(base64.urlsafe_b64decode(img))
    # Reading image and resing to nn input
    input_img = Image.open(input_img).resize((32, 32))
    # Converting to np.array and removing alpha channel
    input_img = np.array(input_img)[:,:,:3]
    # Negate image
    input_img = tf.keras.applications.mobilenet_v2.preprocess_input(input_img)
    return tf.expand_dims(input_img, axis=0)

async def load_saved_model():

    return tf.keras.models.load_model("digits_model_best")

@app.post("/classify")
async def root(img: bytes = File(...), model: tf.keras.Model = Depends(load_saved_model)):
    result = {"prediction" :"Empty", "probability" :{}}

    input_img = preprocess_img(img)
    res = model.predict(input_img)[0]
    
    result["prediction"] = str(np.argmax(res))
    result["probability"] = res[np.argmax(res)] * 100

    return result

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})
