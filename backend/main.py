# backend/main.py

import uuid

import cv2
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
from PIL import Image

import config
import inference
from model import DRModel
import torch
import torch.nn 
import torch.nn.functional as F
app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


@app.post("/pred")
def get_image(file: UploadFile = File(...)):
    image = Image.open(file.file)
    model = DRModel(arch = config.arch)
    output = inference.inference(model, image)
    print(output)
    #name = f"/storage/{str(uuid.uuid4())}.jpg"
    #cv2.imwrite(name, output)
    return {"value": output}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)