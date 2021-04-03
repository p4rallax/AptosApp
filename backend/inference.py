# backend/inference.py

import config
import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np 
from rounder import rounder
def inference(model, image):
    model_path= config.MODEL_PATH
    model = model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #height, width = int(image.shape[0]), int(image.shape[1])
    #new_width = int((640 / height) * width)
    #resized_image  = image.resize((224, 224), resample=Image.BILINEAR)
    #resized_image = np.asarray( resized_image, dtype=np.uint8 )
    #resized_image = transforms.ToTensor()(resized_image)
    
    loader = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    image = loader(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  
    #image.cuda()  

    #model.setInput(inp_blob)
    output = model(image)
    output = rounder(output)
    
    #output = output.transpose(1, 2, 0)
    return output