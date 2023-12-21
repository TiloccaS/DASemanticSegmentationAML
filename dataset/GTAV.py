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
import json

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def pil_loader_label(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
def conta_elementi(tensore):
    # Creazione di un tensore booleano con True per gli elementi che sono maggiori di 19 e diversi da 255
    condizione = (tensore > 19) & (tensore != 255)
    
    # Estrai i valori che soddisfano la condizione
    valori_soddisfacenti = tensore[condizione]
    print("ci sono ", len(valori_soddisfacenti), "sbagliati su ", tensore.numel())
class GtaV(Dataset):
    
    def __init__(self, mode,root):
        super(GtaV, self).__init__()
        # TODO
        self.root = os.path.normpath(root)
        self.split = mode
        images_paths = []
        labels_paths = []
        #si uniscono tutte le directory per imaggini e label in modo da andare a pescare la directory che ci interessa
        image_dir = os.path.join(self.root)
        label_dir = os.path.join(self.root)
        self.root = os.path.normpath(image_dir)
        with open('./dataset/cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        #questo prende l'immagine dal pil e la trasforma in tensore 
        #qui ho un dubbio se occorre normalizzare il tensore
        self.to_tensor = transforms.Compose([
                         transforms.Resize((512, 1024)),
                         transforms.ToTensor(),
                         #To tensor di default trasforma l'immaigne del pil in un tensore con valori che vanno da 0 a 1
                         
                         ])
        self.to_tensor_label = transforms.Compose([
                    transforms.Resize((512, 1024)),

                    transforms.PILToTensor() 
                ])
        #questo trasforma la label in tensore, is usa un compose diverso perche per la label ci serve in scala [0,255] e non [0,1]
        #dubbio
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
        image,label = pil_loader(image_path),Image.open(label_path)

        image=self.to_tensor(image)
        label=self.to_tensor_label(label)
        torch.set_printoptions(profile="full")
        label=self.convert_labels(label)
        conta_elementi(label)

        return image, label   

    def __len__(self):
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label