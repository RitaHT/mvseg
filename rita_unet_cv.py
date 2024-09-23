from sklearn.model_selection import KFold
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
import torch
import numpy as np
print("Start running", flush=True) 

import os
from os import listdir, path
import sys
import time
import numpy as np
#import nibabel as nib #for reading .nii.gz format MRI files
import matplotlib.pyplot as plt #use pyplot in matplotlib instead, pylab is about to archived
import pandas as pd

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torch import from_numpy

import monai
from monai.data import CacheDataset, DataLoader, CSVDataset, Dataset, set_track_meta
from monai.data.utils import pad_list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from torch.nn import BCEWithLogitsLoss
from monai.metrics import DiceMetric
from monai.transforms.transform import Transform, MapTransform
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscreted, AsDiscrete,
    Compose,
    LoadImaged, LoadImage,
    Resized,
    CenterSpatialCropd,
    ScaleIntensityRanged, ScaleIntensityRange,
    NormalizeIntensityd,
    Spacingd,
    EnsureTyped,
    ToTensord,
    RandRotate90d,
    RandFlipd,
    RandAdjustContrastd,
    RandGaussianNoised,
    Rand2DElasticd,
    RandShiftIntensityd,
    SaveImage,
)

""" Directory & Constants: """

#!nvcc --version
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}", flush=True)

# modality
modality = "PET" # PET, CT: used in training loop

# Defining model name
cwd = os.getcwd() #/data/deasy/shiqin_data/intern24/
model_folder = "model_0923"
model_name = f"{modality}_unet"
model_dir = os.path.join(cwd, f"{model_folder}/{model_name}") #
graph_dir = os.path.join(model_dir, "graphs")
if path.exists(model_dir):
    print(f"Found model with same name: {model_dir}")
    response = input("Do you want to use the same name? (yes/no): ").strip().lower()
    if response == "no":
        print("Please change the model_name and rerun") # later add function to change name
        sys.exit()
else:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    print(f"Create new folder {model_dir}")
    print(f"Create new folder {graph_dir}")
bestval_save_dir = os.path.join(model_dir, "model_bestval.pth")
print(f"model_dir: {model_dir}", flush=True)

# Number of folds for K-Fold Cross Validation
k_folds = 5
batch_size = 4
max_epochs = 3

# Load data using the MONAI's decathlon format reader, specifically for Hecktor dataset (ct, pet, label)
file_dir = './'
json_file =  f"{file_dir}/ex_data.json"
datalist = load_decathlon_datalist(json_file, is_segmentation=True, 
                                   data_list_key="image", base_dir="./")
print(len(datalist), flush=True)

# Max, min intensity for transform
a_min_ct = -1024  # 
a_min_pet = 0  # 
a_max_ct = 1000  # Example intensity max value for CT
a_max_pet = 10000  # Example intensity max value for PET

# Define the transforms for training and validation sets
def get_transforms(a_min_ct, a_min_pet, a_max_ct, a_max_pet):
    train_transforms = Compose([
        LoadImaged(keys=["CT", "PET", "Mask"]),
        EnsureChannelFirstd(keys=["CT", "PET", "Mask"]),
        Spacingd(keys=["CT", "PET", "Mask"], pixdim=(1.0, 1.0, 3.0), mode=("bilinear", "bilinear", "nearest"), ensure_same_shape=True),
        ScaleIntensityRanged(keys=["CT"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["PET"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
        CenterSpatialCropd(keys=["CT", "PET", "Mask"], roi_size=(512, 512, 128)),
        Resized(keys=["CT", "PET", "Mask"], spatial_size=(256, 256, 128)),
        # RandFlipd(keys=["CT", "PET", "Mask"], prob=0.5, spatial_axis=0),
        # RandRotate90d(keys=["CT", "PET", "Mask"], prob=0.5, spatial_axes=(0, 1)),
        EnsureTyped(keys=["CT", "PET", "Mask"], device=device),
        ToTensord(keys=["CT", "PET", "Mask"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["CT", "PET", "Mask"]),
        EnsureChannelFirstd(keys=["CT", "PET", "Mask"]),
        Spacingd(keys=["CT", "PET", "Mask"], pixdim=(1.0, 1.0, 3.0), 
                 mode=("bilinear", "bilinear", "nearest"), 
                 ensure_same_shape=True),
        ScaleIntensityRanged(keys=["CT"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys=["PET"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
        CenterSpatialCropd(keys=["CT", "PET", "Mask"], roi_size=(512, 512, 128)),
        Resized(keys=["CT", "PET", "Mask"], spatial_size=(256, 256, 128)),
        EnsureTyped(keys=["CT", "PET", "Mask"], device=device),
        ToTensord(keys=["CT", "PET", "Mask"]),
    ])

    return train_transforms, val_transforms


""" Model: """
# Define UNet model 
class UNet_Segmenter(nn.Module):
    def __init__(self, unet):
        super(UNet_Segmenter, self).__init__()
        self.unet = unet
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs_seg = self.unet(x)  # [batch, 1, 256, 256] if using 2D UNet
        outputs_active = self.sigmoid(outputs_seg)
        return outputs_active

base_model = UNet(
    spatial_dims = 3,  # Use 3D UNet, not slicing
    in_channels = 1,  # PET, CT
    out_channels = 1,  # Binary segmentation
    channels=(64, 128, 256),  # Number of features at each layer #(16, 32, 64, 128, 256)
    strides=(2, 2)  # Downsampling steps
).to(device)

model = UNet_Segmenter(base_model).to(device)

# Define Loss, Optimizer
torch.backends.cudnn.benchmark = True
lr = 0.0005
weight_decay=0.00005
loss_function = DiceCELoss(include_background=True, to_onehot_y=False, sigmoid=False, squared_pred=True) #1.0090 #to_onehot_y=True for [batch,1,3,256,256] #sigmoid already in the model? #why use squared_pred in denominator?

# Define dice metric
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
discrete_pred = AsDiscrete(to_onehot=None, threshold=0.5)  #should be probability #to_onehot=number of classes, threshold=values
discrete_label = AsDiscrete(to_onehot=None, threshold=0.5) #originally not only [0,1] but PROSTATEx docu said it is binarized, bg=0 always

""" Training: """
# Split data using KFold from sklearn
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Main training loop over K-folds
for fold, (train_idx, val_idx) in enumerate(kf.split(datalist)):
    print(f"Training fold {fold+1}/{k_folds}, set to run {max_epochs} epochs", flush=True)

    # Split the dataset
    train_files = [datalist[i] for i in train_idx]
    val_files = [datalist[i] for i in val_idx]

    train_transforms, val_transforms = get_transforms(a_min_ct, a_max_ct, 
                                                      a_min_pet, a_max_pet)

    print("Data loading...", flush=True)
    train_dataset = CacheDataset(data=train_files, transform=train_transforms, 
                                 cache_rate=1.0, progress=False)
    val_dataset = CacheDataset(data=val_files, transform=val_transforms, 
                               cache_rate=1.0, progress=False)
    
    print(f"dataset len: {len(train_dataset)}", flush=True)
    print(f"dataset len: {len(val_dataset)}", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              drop_last=True,
                              collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=pad_list_data_collate)
    
    print(f"dataloader length: {len(train_loader)}", flush=True)
    print(f"dataloader length: {len(val_loader)}", flush=True)

    # Reset model, optimizer, and scaler before each fold
    model = UNet_Segmenter(base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # Training and Validation for each fold
    print("Training...", flush=True)
    best_metric = -1
    best_epoch = -1
    epoch_loss_values = []
    train_dices = []
    val_dices = []
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch_data[f"{modality}"].to(device), batch_data["Mask"].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss) #debug

        # Validation
        model.eval()
        with torch.no_grad():
            train_dice = 0.0
            for train_data in train_loader:
                train_inputs, train_labels = train_data[f"{modality}"].to(device), train_data["Mask"].to(device)
                with torch.cuda.amp.autocast():
                    train_outputs = model(train_inputs)
                dice_metric(y_pred=train_outputs, y=train_labels)
            train_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            train_dices.append(train_dice)

        with torch.no_grad():
            val_dice = 0.0
            for val_data in val_loader:
                val_inputs, val_labels = val_data[f"{modality}"].to(device), val_data["Mask"].to(device)
                with torch.cuda.amp.autocast():
                    val_outputs = model(val_inputs)
                dice_metric(y_pred=val_outputs, y=val_labels)
            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            val_dices.append(val_dice)

        # Save best model for this fold
        if val_dice > best_metric:
            best_metric = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, f"best_model_fold_{fold}.pth"))

        print(f"Epoch {epoch+1}/{max_epochs} - Fold {fold+1}/{k_folds}: Train Loss {epoch_loss:.4f}, Val Dice {val_dice:.4f}", flush=True)
    
    print(f"Best validation dice score for fold {fold+1}: {best_metric:.4f} at epoch {best_epoch}", flush=True)

# Average metrics over all folds
