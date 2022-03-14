import torch
import torchvision

import cv2
import os
import numpy as np

import config
import utils
from models.unet import UNet
from loader import Loader

torch.manual_seed(2022)


#################
# Data loaders
#################
test_dataset = Loader(config.TEST_DIR, "test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
device = torch.device(config.DEVICE)

#################
# Create Model
#################
model = UNet().to(device)
model.load_state_dict(torch.load(config.PRETRAINED_WEIGHTS))
lossF = torch.nn.MSELoss()

#################
# Create output folder
#################
os.makedirs(config.INFERENCE_OUTPUT_DIR, exist_ok=True)

#################
# Inference
#################
print("Start inference")

model.eval()
with torch.no_grad():
    for data, name in test_loader:
        data = data.to(device)
        output = model(data)
        for i in range(0, len(data)):
            utils.save_inference(output[i], os.path.join(config.INFERENCE_OUTPUT_DIR, name[i] + ".png"))
            utils.save_inference(data[i], os.path.join(config.INFERENCE_OUTPUT_DIR, name[i] + "_ref.png"))
            utils.save_anomaly_overlay(data[i].cpu().numpy(), output[i].cpu().numpy(), os.path.join(config.INFERENCE_OUTPUT_DIR, name[i] + "_thresh.png"))