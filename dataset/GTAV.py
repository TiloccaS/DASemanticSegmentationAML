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
from torchvision.transforms import v2
import torch
import numpy as np
import json
from dataset.utils import pil_loader


class GtaV(Dataset):
    
    def __init__(self, root, aug_type,height,width):
        super(GtaV, self).__init__()
        # TODO
        self.root = os.path.normpath(root)
        images_paths = []
        labels_paths = []
        self.resize=(height,width)
        image_dir = os.path.join(self.root)
        self.root = os.path.normpath(image_dir)
        #the classes of the gt images are 34, we need to pass from 34->19, so we need a mapping
        with open('./dataset/gta5_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
       
            
        normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        bright_t = transforms.ColorJitter(brightness=[1,2])
        contrast_t = transforms.ColorJitter(contrast = [2,5])
        saturation_t = transforms.ColorJitter(saturation = [1,3])
        hue_t = transforms.ColorJitter(hue = 0.2)
        gs_t = transforms.Grayscale(3)
        hflip_t = transforms.RandomHorizontalFlip(p = 1)
        rp_t = transforms.RandomPerspective(p = 1, distortion_scale = 0.5)
        rot_t = transforms.RandomRotation(degrees = 90)

        aug_transformations = {
            "CS-HF": transforms.Compose([contrast_t, saturation_t, hflip_t]),
            "H-RP": transforms.Compose([hue_t, rp_t]),
            "B-GS-R": transforms.Compose([bright_t, gs_t, rot_t])
            }

        if aug_type is not None:
            aug_transformation = aug_transformations[aug_type]
            self.aug_pipeline = transforms.Compose([
                                                transforms.RandomApply([aug_transformation], p = 0.5),
                                                transforms.ToTensor(),
                                                normalizer
                                                ])
        else:
            self.aug_pipeline = transforms.Compose([
                    transforms.ToTensor(),
                    normalizer
            ])

     
        self.to_tensor_label = transforms.Compose([

                    transforms.PILToTensor() 
                ])
        
        image_files = os.listdir(os.path.normpath(os.path.join(self.root, 'images')))

        image_files.sort()

        
        for img in image_files:
            images_paths.append(self.root+'/images/'+img)
        
        label_files = os.listdir(os.path.normpath(os.path.join(self.root, 'labels')))


        label_files.sort()

   
        for lbl in label_files:
           labels_paths.append(self.root+'/labels/'+lbl)
           
           
        label_order=sorted(labels_paths)
        image_order=sorted(images_paths)
        self.data = pd.DataFrame(zip(image_order, label_order), columns=["image_path", "label_path"])




    def __getitem__(self, idx):
        image_path = self.data["image_path"].iloc[idx]
        label_path=self.data["label_path"].iloc[idx]
        image,label = pil_loader(image_path),Image.open(label_path)
        image=image.resize(self.resize,Image.BILINEAR)
        label=label.resize(self.resize,Image.NEAREST)
        image=self.aug_pipeline(image)
        label=self.to_tensor_label(label)
        label=self.convert_labels(label)
        return image, label   

    def __len__(self):
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label