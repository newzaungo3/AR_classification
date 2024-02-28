import udpFucn as U
import os
from PIL import Image, ImageFile
import math
import copy
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import torch.onnx
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.transforms import Resize, CenterCrop, Pad, Compose
import torchvision.transforms.functional as F
import open_clip

sock = U.UdpComms(
    udpIP="127.0.0.1", portTX=5000, portRX=7000, enableRX=True, suppressWarnings=True
)

model_name = 'coca_ViT-L-14' #'ViT-L-14-CLIPA'
pretrained = 'mscoco_finetuned_laion2b_s13b_b90k' #'datacomp1b'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(model_name,
                                                             device = device,
                                                             pretrained=pretrained)

# Import tensor from pickle file
with open('tensor.pkl', 'rb') as f:
    q_embeddings = pickle.load(f)
with open('class.pkl', 'rb') as d:
    query_classes = pickle.load(d)

# Print the loaded tensor
# print(q_embeddings.shape)
# print(query_classes)

def main():
    print("hello")
    while True:
        try:
            data = sock.ReadReceivedData()
            if (data != None):
                print(data)
                print(type(data))
                
                data = data.convert('RGB')
                inputs = preprocess(data).to(device)
                inputs = inputs.unsqueeze(0)
                print(inputs.shape)
                
                outputs = model.encode_image(inputs).cpu()
                outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
                print(outputs.shape)
                
                time.sleep(10)
                sock.SendData("Hello from python")
            
        except WindowsError as e:
                print(e)
                break    
      
      
if __name__ == "__main__":
    main()
  