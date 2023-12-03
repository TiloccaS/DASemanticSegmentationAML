#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from PIL import Image

import os
import os.path
import sys
import pandas as pd
import torchvision
from torchvision import transforms
import torch
import numpy as np


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def pil_loader_label(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
class GtaV(Dataset):
    
    def __init__(self, mode,root):
        super(GtaV, self).__init__()
        # TODO
        self.root = os.path.normpath(root)
        self.split = mode
        images_paths = []
        labels_paths = []
        image_dir = os.path.join(self.root)
        label_dir = os.path.join(self.root)
        self.root = os.path.normpath(image_dir)
        self.transform = transforms.Compose([transforms.Resize(256),      # Resizes short size of the PIL image to 256
                                             transforms.CenterCrop(224),  # Crops a central square patch of the image
                                                                   # 224 because torchvision's AlexNet needs a 224x224 input!
                                                                    # Remember this when applying different transformations, otherwise you get an error
                                             transforms.ToTensor()])

        image_files = os.listdir(os.path.normpath(os.path.join(self.root, 'images')))

        image_files.sort()

        split_index = int(0.75 * len(image_files)) 

        split_files = image_files[:split_index] if self.split == 'train' else image_files[split_index:]
        for img in split_files:
            images_paths.append(self.root+'/images/'+img)
        
        label_files = os.listdir(os.path.normpath(os.path.join(self.root, 'labels')))


        label_files.sort()

        split_index = int(0.75 * len(image_files)) 
        split_labels = label_files[:split_index] if self.split == 'train' else label_files[split_index:]

        for lbl in split_labels:
           labels_paths.append(self.root+'/labels/'+lbl)
           

        self.data = pd.DataFrame(zip(images_paths, labels_paths), columns=["image_path", "label_path"])


    def __getitem__(self, idx):
        image_path = self.data["image_path"].iloc[idx]
        label_path=self.data["label_path"].iloc[idx]
        image, label = pil_loader(image_path), pil_loader_label(label_path)

        image=self.transform(image)
        label = self.transform(label)
        return image, label

    def __len__(self):
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
     