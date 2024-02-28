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

sock = U.UdpComms(
    udpIP="127.0.0.1", portTX=5000, portRX=7000, enableRX=True, suppressWarnings=True
)

