#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image,ImageFile
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
from dataset.utils import pil_loader

class CityScapes(Dataset):
    
    def __init__(self, mode,root,height,width):
        super(CityScapes, self).__init__()
        # TODO
        self.root = os.path.normpath(root)
        self.split = mode
        images_paths = []
        labels_paths = []
        normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.resize=(height,width)
        image_dir = os.path.join(self.root, 'images', mode)
        label_dir = os.path.join(self.root, 'gtFine', mode)
        self.root = os.path.normpath(image_dir)
        
        #transform the images in a tensor
        self.to_tensor = transforms.Compose([
                         transforms.ToTensor(),
                         normalizer
                         #To tensor di default trasforma l'immaigne del pil in un tensore con valori che vanno da 0 a 1
                         
                         ])
        #This transforms the label into a tensor, is uses a different component because for the label we need to scale [0,255] and not [0,1]

        self.to_tensor_label = transforms.Compose([
                    #transforms.Resize((2048, 1024)),

                    transforms.PILToTensor() 
                ])
       
        
        for city in os.listdir(image_dir):
            folder_path = os.path.join(image_dir, city)
            #dataset is splitted by city, so e need to change directory
            if os.path.isdir(folder_path):
                 for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(folder_path, filename)
                        images_paths.append(image_path)
        
        for city_label in os.listdir(label_dir):
            folder_path = os.path.join(label_dir, city_label)
            if os.path.isdir(folder_path):
                 for filename in os.listdir(folder_path):
                    #to use only label with 19 classes 
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and 'color' not in filename.lower():
                        label_path = os.path.join(folder_path, filename)
                        labels_paths.append(label_path)
        
        #sorting we can assign to each images the correspondance label    
        label_order=sorted(labels_paths)
        image_order=sorted(images_paths)
        self.data = pd.DataFrame(zip(image_order, label_order), columns=["image_path", "label_path"])


    def __getitem__(self, idx):
        image_path = self.data["image_path"].iloc[idx]
        label_path=self.data["label_path"].iloc[idx]
        image,label = pil_loader(image_path),Image.open(label_path)
        image=image.resize(self.resize,Image.BILINEAR)
        label=label.resize(self.resize,Image.NEAREST)
        image=self.to_tensor(image)
        label=self.to_tensor_label(label)
        return image, label
    

    


    def __len__(self):
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
     