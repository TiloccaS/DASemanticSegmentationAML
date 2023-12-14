#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image
from utils import one_hot_it_v11,RandomCrop 
import os
import os.path
import sys
import pandas as pd
import torchvision
from torchvision import transforms
import torch
import numpy as np
import json
import random
import cv2

def cv2_loader_label(path):
    # Carica l'immagine in scala di grigi
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

def cv2_loader(path):
    # Carica l'immagine a colori
    img = cv2.imread(path)
    # Converti da BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
class CityScapes(Dataset):
    
    def __init__(self, mode,root, cropsize=(640, 480),randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5)):
        super(CityScapes, self).__init__()
        # TODO
        self.root = os.path.normpath(root)
        self.split = mode
        images_paths = []
        labels_paths = []
        image_dir = os.path.join(self.root, 'images', mode)
        label_dir = os.path.join(self.root, 'gtFine', mode)
        self.root = os.path.normpath(image_dir)
        self.to_tensor = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                         ])
        self.to_tensor_label = transforms.Compose([
                 # other transforms
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0,)*3, std=(255,)*3)
                        ])
        
        for city in os.listdir(image_dir):
            folder_path = os.path.join(image_dir, city)
            if os.path.isdir(folder_path):
                 for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(folder_path, filename)
                        images_paths.append(image_path)
        
        for city_label in os.listdir(label_dir):
            folder_path = os.path.join(label_dir, city_label)
            if os.path.isdir(folder_path):
                 for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and 'color' not in filename.lower():
                        label_path = os.path.join(folder_path, filename)
                        labels_paths.append(label_path)
           

        self.data = pd.DataFrame(zip(images_paths, labels_paths), columns=["image_path", "label_path"])


    def __getitem__(self, idx):
        image_path = self.data["image_path"].iloc[idx]
        label_path=self.data["label_path"].iloc[idx]
        image,label = cv2_loader(image_path),cv2_loader_label(label_path)

        image=self.to_tensor(image).float()
        label=self.to_tensor_label(label).float()

        return image, label
    

    def convert_labels(self, label):
        for k, v in self.map_label.items():
            label[label == k] = v
        return label
  


    def __len__(self):
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
     