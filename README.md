# DASemanticSegmentationAML
Semantic segmentation is a crucial task for imageanalysis, but its effectiveness can be compromised when the modelis applied to a domain other than the one in which it was trained. What we propose in this repository is the possibility to do Semantic Segmentation and Domain Adaptation on datasets such as Cityscapes and GTAV, moreover we propose the use of the [NNI](https://nni.readthedocs.io/en/stable/) tool capable of otimizing hyperparameters.

## Background
Semantic segmentation is a classic and fundamental topic in computer vision, which aims to assign pixel-level labels in images. The prosperity of deep learning greatly promotes the performance of semantic segmentation by making various breakthroughs, coming with fast-growing demands in many applications, e.g., autonomous driving, video surveillance, robot sensing, and so on. Semantic Segmentation is a crucial task for image analysis, but its effectiveness can be compromised when the model is applied to a domain other than the one in which it was trained, Domain Adaptation is a technique that can help us in these cases, they have been developed to address the domain-shift problem between the source and target domains. The main insight behind these approaches is to tackle the problem by aligning the feature distribution between source and target images.

## Getting started
### Data Download
The dataset of GTAV and Cityscapes are available [here](https://drive.google.com/drive/folders/1iE8wJT7tuDOVjEBZ7A3tOPZmNdroqG1m)

### Installation
This code has been tested on python 3.9.16.
```

git clone https://github.com/TiloccaS/DASemanticSegmentationAML.git
pip install --upgrade pip
pip install -r requirements.txt

```
## How to Use the Project:
### 2nd TESTING REAL-TIME SEMANTIC SEGMENTATION

####  A) Defining the upper bound for the domain adaptation phase
```
python train.py --root <ROOT PATH CITYSCAPES> --pretrain_path './pretrained_models/STDCNet813M_73.91.tar' --backbone 'STDCNet813' --save_model_path './Cityscapes_model  --num_epochs 50 --batch_size 8 
--num_workers 4 --

```
####  B)  Train on synthetic datasets
```
python train.py --root <ROOT PATH GTAV> --pretrain_path './pretrained_models/STDCNet813M_73.91.tar' --dataset GTAV --backbone 'STDCNet813' --save_model_path './GTA5_model' --num_epochs 50 --batch_size 8 --num_workers 4 

```

#### C) Evaluate the domain shift.
Without Data Augmentation:


```
python train.py --root <ROOT PATH CITYSCAPES> --pretrain_path './GTA5_model/NoDA_best.pth' --backbone 'STDCNet813' --save_model_path './GTA5_model' --domain_shift True

```

With Data Augmentation: 

```
python train.py --root <ROOT PATH CITYSCAPES> --pretrain_path './GTA5_model/H-RP_best.pth' --backbone 'STDCNet813' --save_model_path './GTA5_model' --domain_shift True

```

### 3nd  IMPLEMENTING UNSUPERVISED ADVERSARIAL DOMAIN ADAPTATION
```
python train.py --root <ROOT PATH CITYSCAPES> --root_source <ROOT PATH GTAV> --root_target <ROOT PATH CITYSCAPES> --pretrain_path './pretrained_models/STDCNet813M_73.91.tar' --backbone 'STDCNet813' --save_model_path './GTA5_model' --aug_type 'H-RP' --domain_adaptation True  --num_epochs 50 --batch_size 8 --num_workers 4 

```
### 4nd  IMPROVEMENTS

#### A) Different and lighter discriminator function with Depth Wise Discriminator

```
python train.py --root <ROOT PATH CITYSCAPES> --root_source <ROOT PATH GTAV> --root_target <ROOT PATH CITYSCAPES> --pretrain_path './pretrained_models/STDCNet813M_73.91.tar' --backbone 'STDCNet813' --save_model_path './GTA5_model' --aug_type 'H-RP' --domain_adaptation True  --num_epochs 50 --batch_size 8 --num_workers 4 --depthwise True

```
#### D) Hyper-parameter optimization to improve results
```
python experiment_nni.py --root <ROOT PATH CITYSCAPES> --root_source <ROOT PATH GTAV> --root_target <ROOT PATH CITYSCAPES> --pretrain_path './pretrained_models/STDCNet813M_73.91.tar' --backbone 'STDCNet813' --save_model_path './GTA5_model_nni' --aug_type 'H-RP' --domain_adaptation True  --optimizer 'sgd'

```

## Credits
### Authors
- Salvatore Tilocca s305938
- Davide Natale s318967
- Salvatore Cabras s320096

### References
- [Rethinking BiSeNet For Real-time Semantic Segmentation](https://github.com/MichaelFan01/STDC-Seg/tree/master)
- [Learning to Adapt Structured Output Space for Semantic Segmentation](https://github.com/wasidennis/AdaptSegNet/tree/master?tab=readme-ov-file)



