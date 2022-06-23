import torch
import torchvision

import os
import cv2
import csv
import natsort
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import config
import utils
from models.unet import UNet
from loader import Loader

torch.manual_seed(2022)

#################
# Data loaders
#################
train_dataset = Loader(config.TRAIN_DIR, "train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE)

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
copyfile("config.py", os.path.join(config.INFERENCE_OUTPUT_DIR, "config.py"))

model.eval()
anomaly_scores = {}
feature_train_table = []

print("Beginning feature extraction for training")
# Training data
with torch.no_grad():
    for data, name in train_loader:
        data = data.to(device)
        output = model.forward_features(data).cpu()
        feature_train_table.append(output)
feature_train_table = torch.cat(feature_train_table)

print("Kmeans training")
kmeans = KMeans(n_clusters=config.KMEANS_CLUSTERS, random_state=0).fit(feature_train_table)

print("Beginning feature extraction for testing")
plot_dict = {}
# Test data
with torch.no_grad():
    for data, names in test_loader:
        data = data.to(device)
        output = model.forward_features(data).cpu()
        #names = [name[j][i] for j in range(len(name)) for i in range(len(name[j]))] 
        pred = []
        for k in range(len(output)) :
            pred.append(kmeans.transform(output[k].unsqueeze(0)).min(axis=1).item()) 
        for i, name in enumerate(names) :
            plot_dict[name] = pred[i]

sorted_dict = dict(natsort.natsorted(plot_dict.items()))
# Write score in csv
with open(os.path.join(config.INFERENCE_OUTPUT_DIR, "distances.csv"), 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_name","distance"])
    for k, v in sorted_dict.items():
        writer.writerow([k, v])




