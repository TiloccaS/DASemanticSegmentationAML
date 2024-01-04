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
#questo lìho aggiunto per un bug
ImageFile.LOAD_TRUNCATED_IMAGES=True
def pil_loader_label(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
 
def modifica_tensore(tensore):
    # Crea una maschera per gli elementi > 19 e diversi da 255
    maschera = (tensore >= 19) & (tensore != 255)

    # Modifica gli elementi che soddisfano la condizione
    tensore[maschera] = 255
    return tensore
def conta_elementi(tensore):
    # Creazione di un tensore booleano con True per gli elementi che sono maggiori di 19 e diversi da 255
    condizione = (tensore > 19) & (tensore != 255)
    
    # Estrai i valori che soddisfano la condizione
    valori_soddisfacenti = tensore[condizione]
    print("ci sono ", len(valori_soddisfacenti), "sbagliati su ", tensore.numel())
class CityScapes(Dataset):
    
    def __init__(self, mode,root, cropsize=(640, 480),randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5)):
        super(CityScapes, self).__init__()
        # TODO
        self.root = os.path.normpath(root)
        self.split = mode
        images_paths = []
        labels_paths = []
        normalizer = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        #si uniscono tutte le directory per imaggini e label in modo da andare a pescare la directory che ci interessa
        image_dir = os.path.join(self.root, 'images', mode)
        label_dir = os.path.join(self.root, 'gtFine', mode)
        self.root = os.path.normpath(image_dir)
        #questo prende l'immagine dal pil e la trasforma in tensore 
        #qui ho un dubbio se occorre normalizzare il tensore
        self.to_tensor = transforms.Compose([
                         transforms.ToTensor(),
                         normalizer
                         #To tensor di default trasforma l'immaigne del pil in un tensore con valori che vanno da 0 a 1
                         
                         ])
        #questo trasforma la label in tensore, is usa un compose diverso perche per la label ci serve in scala [0,255] e non [0,1]
        #dubbio
        self.to_tensor_label = transforms.Compose([
                    #transforms.Resize((2048, 1024)),

                    transforms.PILToTensor() 
                ])
       
        
        for city in os.listdir(image_dir):
            folder_path = os.path.join(image_dir, city)
            #se andate a vedere come è fatto il dataset è diviso per citta quindi occorre prima spostarci in quella cartella e poi salvare l'immagine
            if os.path.isdir(folder_path):
                 for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(folder_path, filename)
                        images_paths.append(image_path)
        
        for city_label in os.listdir(label_dir):
            folder_path = os.path.join(label_dir, city_label)
            if os.path.isdir(folder_path):
                 for filename in os.listdir(folder_path):
                    #check per vedere se la parola colo non è in file name perche a noi interessano solo le immagini in bianco e nero
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and 'color' not in filename.lower():
                        label_path = os.path.join(folder_path, filename)
                        labels_paths.append(label_path)
           

        self.data = pd.DataFrame(zip(images_paths, labels_paths), columns=["image_path", "label_path"])


    def __getitem__(self, idx):
        image_path = self.data["image_path"].iloc[idx]
        label_path=self.data["label_path"].iloc[idx]
        image,label = pil_loader(image_path),Image.open(label_path)
        image=image.resize((512,1024),Image.BILINEAR)
        label=label.resize((512,1024),Image.NEAREST)

        image=self.to_tensor(image)
        label=self.to_tensor_label(label)
        torch.set_printoptions(profile="full")
        #print(label)
        #label=modifica_tensore(label)
        #conta_elementi(label)
        return image, label
    

    


    def __len__(self):
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
     