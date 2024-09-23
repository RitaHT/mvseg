# -*- coding: utf-8 -*-
"""Rita MobileViT playground more channels.ipynb

Original file is located at https://colab.research.google.com/drive/1thmUo66sG8l9ooRgfLQGajzTB-OjD-oA
A copy on GitHub: https://github.com/RitaHT/mvseg

### MobileViT Playground
MobileViT Paper: https://arxiv.org/abs/2110.02178 (2021)
DeepLabV3 Paper: https://arxiv.org/abs/1706.05587 (2017)
Moco self-supervised pretraining: https://arxiv.org/abs/1911.05722 (2019)

Goal:
*   MobileViT w/ given pretrained weights
"""

print("Start running", flush=True) 

import os
from os import listdir, path
import sys
import time
import json
from sklearn.model_selection import train_test_split
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
from monai.data import CacheDataset, DataLoader, CSVDataset, Dataset, set_track_meta, load_decathlon_datalist
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
    RandRotated,
    RandFlipd,
    RandAdjustContrastd,
    RandGaussianNoised,
    Rand2DElasticd,
    RandShiftIntensityd,
    SaveImage,
)

print("all packages imported", flush=True)


""" Functions: """

def validation(model, val_loader, dice_metric, device):
    # evaluation loop
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            inputs_v, labels_v = (batch_data[f"{modality}"].to(device), batch_data["Mask"].to(device),)# [batch, 3, 256, 256]
            # print(f"x shape: {inputs_v.shape}")
            # print(f"y shape: {labels_v.shape}")
            with torch.cuda.amp.autocast():
                outputs_v = model(inputs_v)
                # print(f"y_pred shape: {outputs_v.shape}", flush=True)
                outputs_v = discrete_pred(outputs_v) 
                labels_v = discrete_label(labels_v)
            #debug
            # print(f"y shape: {labels_v.shape}")
            # print(f"y_pred shape: {outputs_v.shape}")
            dice_metric(y_pred=outputs_v, y=labels_v) #num_class=shape[1]
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    #print(f"Evaluation finished! Average loss = {avg_loss:2.5f}")
    model.train() #do i need this???
    return mean_dice_val

def export_ex_seg(batch_data, model_name, graph_dir, val=True):
    # model.eval()
    inputs, labels = (batch_data[f"{modality}"], batch_data["Mask"]) #no ID yet?
    outputs_upsampled = model(inputs)
    # model.train() # do i need to do this???
    outputs_upsampled = outputs_upsampled.detach() #grad
    outputs_upsampled_v = discrete_pred(outputs_upsampled)
    if outputs_upsampled_v.shape[1:] != (1, 256, 256, 128): #do not print, when normal
        print(f"output shape: {outputs_upsampled_v.shape}")
    labels_v = discrete_label(labels_v)
    if labels_v.shape[1:] != (1, 256, 256): #do not print, when normal
        print(f"label  shape: {labels_v.shape}")

    i = 0 #change if needed, batch_size=1/4
    dice_metric(y_pred=outputs_upsampled_v, y=labels_v) #compute dice for one image in this batch #num_class=shape[1] #must be [i,:,...]=[batch,channel,spatial]
    example_dice= dice_metric.aggregate().item() # 0.004
    dice_metric.reset()
    # print(f"example_dice: {example_dice}")

    results_upsampled = outputs_upsampled.cpu().detach().numpy()
    inputs_for_eval = inputs.cpu()
    labels_for_eval = labels.cpu()

    threshold = 0.5
    result_masks = (results_upsampled > threshold) #[batch,1,256,256]

    fig = plt.figure(figsize=(5, 5))
    #assume 3 same slices
    a = fig.add_subplot(2, 2, 1); a.set_title("original image", fontsize=8); a.axis("off");
    plt.imshow(np.rot90(inputs_for_eval[i,1,:,:]), cmap="gray")

    a = fig.add_subplot(2, 2, 3); a.set_title("original mask", fontsize=8);a.axis("off");
    plt.imshow(np.rot90(labels_for_eval[i,1,:,:]), cmap="gray")

    a = fig.add_subplot(2, 2, 4);a.set_title("Interp output(mask)", fontsize=8);a.axis("off");
    plt.imshow(np.rot90(result_masks[i,0,:,:]))

    a = fig.add_subplot(2, 2, 2); a.set_title("Overlay", fontsize=8); a.axis("off")
    plt.imshow(np.rot90(inputs_for_eval[i,1,:,:]), cmap="gray") #middle slice
    plt.contour(np.rot90(labels_for_eval[i,1,:,:]),  colors='red', linewidths = 0.6)
    plt.contour(np.rot90(result_masks[i,0,:,:]),  colors='lime', linewidths = 0.8) #[batch, 1, 256,256] output #result must be 2D, use [i,0,..] not [i,:,..]

    if val:
        fig.suptitle(f"val_dataset {ID[i]} \n example_dice_coeff = {example_dice}",fontsize=10)
        plt.savefig(f'{graph_dir}/ex_val_{model_name}.png')
    else:
        fig.suptitle(f"train_dataset {ID[i]} \n example_dice_coeff = {example_dice}",fontsize=10)
        plt.savefig(f'{graph_dir}/ex_train_{model_name}.png')
    plt.close()

    return example_dice

def export_ex_seg_train_val(model_name, graph_dir, 
                            train_loader, val_loader, epoch,
                            train_i_stop = 0, val_i_stop = 6):
    print(f"Exporting eximage on Training Set", flush=True)
    train_i = 0
    for batch_data in train_loader:
        train_i += 1
        if train_i == train_i_stop: #change if needed
            break
    export_ex_seg(batch_data, f"{model_name}_epoch{epoch}", graph_dir, val=False)
    
    print(f"Exporting eximage on Validation Set", flush=True)
    val_i = 0
    for batch_data in val_loader: #batchsize=4
        val_i += 11 #ProstateX-0092
        if val_i == val_i_stop: #change if needed#ProstateX-0169-Finding1_slices_0
            break
    export_ex_seg(batch_data, f"{model_name}_epoch{epoch}", graph_dir, val=True)


""" Directory & Constants: """

#!nvcc --version
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}", flush=True)

# Defining model name
modality = "CT" #PET, CT,  #used in dataloading below
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

# constants
max_epochs = 10  #50

# load 
print("Data loading...", flush=True)
file_dir = './'


""" ### Put into dataset using MONAI """
"""#### Load"""

# Load HECKTOR JSON data
hecktor_data_file = f'{file_dir}/ex_data.json'  # train_json.json #ex_data.json
datalist = load_decathlon_datalist(hecktor_data_file, is_segmentation=True, 
                                   data_list_key="image", base_dir="./")

# with open(hecktor_data_file) as f:
#     hecktor_data = json.load(f)

# # Parse JSON to get "ct", "pet", "label" paths
# image_files = []
# label_files = []
# for case in hecktor_data['image']:
#     # print(case)
#     image_files.append({"ct": case["CT"], "pet": case["PET"]})
#     label_files.append(case["Mask"])

# # Split into train/val (80:20)
# train_images, val_images, train_labels, val_labels = train_test_split(
#     image_files, label_files, test_size=0.2, random_state=42
# )

train_files, val_files = train_test_split(
    datalist, test_size=0.2, random_state=42
)

# print(val_files) #{'CT': 'imagesTr/CHUM-013__CT.nii.gz', 'PET': 'imagesTr_PET_aligned_norm_0_1/CHUM-013__PT.nii.gz', 'Mask': 'labelsTr/CHUM-013.nii.gz'},

# Define transforms
a_min_ct = -1024  # 
a_min_pet = 0  # 
a_max_ct = 1000  # Example intensity max value for CT
a_max_pet = 10000  # Example intensity max value for PET

train_transforms = Compose([
    LoadImaged(keys=["CT", "PET", "Mask"]),
    EnsureChannelFirstd(keys=["CT", "PET", "Mask"]),
    Spacingd(keys=["CT", "PET", "Mask"], pixdim=(1.0, 1.0, 3.0), 
             mode=("bilinear", "bilinear", "nearest"),
             ensure_same_shape=True
             ),
    ScaleIntensityRanged(keys=["CT"], a_min=a_min_ct, a_max=a_max_ct, b_min=0.0, b_max=1.0, clip=True),
    ScaleIntensityRanged(keys=["PET"], a_min=a_min_pet, a_max=a_max_pet, b_min=0.0, b_max=1.0, clip=True),
    CenterSpatialCropd(keys=["CT", "PET", "Mask"], roi_size=(512,512,128)),
    Resized(keys=["CT", "PET", "Mask"], spatial_size=(256, 256, 128)), #-1 #hardcoded to 110
    EnsureTyped(keys=["CT", "PET", "Mask"]),
    RandRotated(keys=["CT", "PET", "Mask"], prob=0.5, range_x=[-1.0, 1.0]),
    # RandAdjustContrastd(keys=["CT"], prob=0.5, gamma=(0.5, 2.0)),
    # RandGaussianNoised(keys=["CT"], prob=0.5, mean=0.0, std=0.1),
    RandFlipd(keys=["CT", "PET", "Mask"], prob=0.5, spatial_axis=1),
    # Rand2DElasticd(keys=["CT", "PET", "Mask"], prob=0.5, spacing=(20, 20), magnitude_range=(1, 1)),
    # RandShiftIntensityd(keys=["CT"], prob=0.5, offsets=(0.1, 0.2)),
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
    CenterSpatialCropd(keys=["CT", "PET", "Mask"], roi_size=(512,512,128)),
    Resized(keys=["CT", "PET", "Mask"], spatial_size=(256, 256, 128)),  #-1 #hardcoded to 120
    EnsureTyped(keys=["CT", "PET", "Mask"]),
    ToTensord(keys=["CT", "PET", "Mask"]),
])

# Create datasets and dataloaders
train_dataset = CacheDataset(
    # data=[{"ct": item["ct"], "pet": item["pet"], "label": label} for item, label in zip(train_images, train_labels)],
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,  # 1.0 #0.01
    progress=False
)
val_dataset = CacheDataset(
    # data=[{"ct": item["ct"], "pet": item["pet"], "label": label} for item, label in zip(val_images, val_labels)],
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0, # 1.0
    progress=False
)

print(f"dataset len: {len(train_dataset)}", flush=True)
print(f"dataset len: {len(val_dataset)}", flush=True)
print(f"dataset img: {train_dataset[0][f'{modality}'].shape}", flush=True)

# example images
val_data_example = train_dataset[2] #val_dataset
plt.ioff()
print(f"image shape: {val_data_example[f'{modality}'].shape}")
plt.figure("image", (24, 6))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.title(f"image {modality} slice {i*10}")
    plt.imshow(np.rot90(val_data_example[f"{modality}"][0, :, :, i*10].detach().cpu()), cmap="gray")
plt.savefig('eximage.png')
plt.close()

# example labels
print(f"label shape: {val_data_example['Mask'].shape}")
plt.figure("label", (24, 6))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.title(f"label slice {i*10}")
    plt.imshow(np.rot90(val_data_example["Mask"][0, :, :, i*10].detach().cpu()))
plt.savefig('exlabel.png')
plt.close()

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          drop_last=True, collate_fn=pad_list_data_collate)# num_workers=2, pin_memory=True)#, collate_fn=pad_list_data_collate) #(239) #still error?? #collate_fn=pad_list_data_collate
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=pad_list_data_collate)# num_workers=2, pin_memory=True) #(60)
#debug:
train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                               drop_last=True, collate_fn=pad_list_data_collate)

print(f"dataloader length: {len(train_loader)}")

i = 0
for batch_data in train_loader:
    i += 1
    if i == 1: #choose to see which example images
        break
input = batch_data[f"{modality}"]
print(f"dataloader img: {input.shape}", flush=True) # [4, 1, 256, 256, 124], prev: [4,3,256,256]
print("Data loaded", flush=True)



"""### Try Training """

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

# Define UNet model
base_model = UNet(
    spatial_dims = 3,  # Use 3D UNet
    in_channels = 1,  # PET, CT #2?
    out_channels = 1,  # Binary segmentation
    channels=(64, 128, 256),  # Number of features at each layer #(16, 32, 64, 128, 256)
    strides=(2, 2)  # Downsampling steps
).to(device)

model = UNet_Segmenter(base_model).to(device)

# Define Loss, Optimizer
torch.backends.cudnn.benchmark = True
lr = 0.0005
weight_decay=0.00005
#loss_function = BCEWithLogitsLoss()   #nn.CrossEntropyLoss()
loss_function = DiceCELoss(include_background=True, to_onehot_y=False, sigmoid=False, squared_pred=True) #1.0090 #to_onehot_y=True for [batch,1,3,256,256] #sigmoid already in the model? #why use squared_pred in denominator?
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) #lr > weight_decay #as paper, base_model & seg_model?
scaler = torch.cuda.amp.GradScaler()
#scheduler?

# Define dice metric
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
discrete_pred = AsDiscrete(to_onehot=None, threshold=0.5)  #should be probability #to_onehot=number of classes, threshold=values
discrete_label = AsDiscrete(to_onehot=None, threshold=0.5) #originally not only [0,1] but PROSTATEx docu said it is binarized, bg=0 always


"""#### Train"""
# training loop
def train(model, train_loader, loss_function, optimizer, scaler, device, 
          model_dir, bestval_save_dir,
          metric_value_best, epoch_loss_value_best, global_step_best,
          global_step_start, max_epochs, num_example, ex_dices, resume=False):

    start_time = time.time()

    num_example = num_example
    global_step = global_step_start
    ori_resolution = (256, 256, 128)
    #debug
    spikes_epochs = []

    print(f"Training starts! Set to run {max_epochs} epochs", flush=True)
    for epoch in range(global_step_start, max_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader: 
            optimizer.zero_grad()
            inputs, labels = (batch_data[f"{modality}"].to(device),batch_data["Mask"].to(device),)
            #debug
            # print(f"inputs shape: {inputs.shape}")
            # print(f"labels shape: {labels.shape}")
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs = model(inputs) #[batch,1,256,256]
                # print(f"outputs shape: {outputs.shape}")
                loss = loss_function(outputs, labels) 
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) #when amp
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()


        epoch_loss /= num_example #average loss over this epoch
        epoch_loss_values.append(epoch_loss)

        # Detect spikes
        # if len(epoch_loss_values) > 1 and (epoch_loss_values[-1] > epoch_loss_values[-2] * 1.05):  # 5% increase from the previous loss
        #     print(f"Spike detected at epoch {epoch} in loss: {epoch_loss_values[-1]}")
        #     spikes_epochs.append(epoch)
        #     print(f"Exporting example images at spike epoch {epoch}")
        #     export_ex_seg_train_val(model_name, graph_dir, 
        #                             train_loader, val_loader, f"{epoch}_spikes",
        #                             train_i_stop = 0, val_i_stop = 6)

        # check to make sure the model is running
        if (epoch <= 40): #hardcoded
            print(f"Training {epoch + 1} epoch (loss={epoch_loss:2.5f})", flush=True)

        # Validation & Save Model
        val_interval = 10 #10
        if (epoch % val_interval == 0 and epoch > 0) or epoch + 1 == max_epochs: #hardcoded to 10, 20 iter, may change later
            model.eval()
            # Training accuracy
            #debug
            # print("Train dice:", flush=True)
            mean_dice_train = validation(model, train_loader, dice_metric, device)
            metric_values_train.append(mean_dice_train)
            # Validation accuracy
            # print("Val dice:", flush=True)
            mean_dice_val = validation(model, val_loader, dice_metric, device)
            metric_values.append(mean_dice_val)
            # Save Model: best validation loss
            if mean_dice_val > metric_value_best: #using MONAI DiceMetric(): larger, better overlap
                metric_value_best = mean_dice_val
                global_step_best = epoch
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': epoch,
                    'best_epoch': global_step_best,
                    'epoch_loss_values': epoch_loss_values,
                    'metric_values': metric_values,
                    'metric_values_train': metric_values_train,
                    'best_loss_values': epoch_loss_value_best,
                    'best_metric_values': metric_value_best,
                }, bestval_save_dir)
                print("Model saved!" , flush=True)
            else:
                print("Model NOT saved!", flush=True)
            
            # get intermediate example images
            # # debug: use train_eval_loader, instead of train_loader
            train_i = 0
            for batch_data in train_eval_loader:
                train_i += 1
                if train_i == 0: #change if needed
                    break
            export_ex_seg(batch_data, f"{model_name}_epoch{epoch}", graph_dir, val=False)
            val_i = 0
            for batch_data in val_loader:
                val_i += 1
                if val_i == 0: #change if needed
                    break
            export_ex_seg(batch_data, f"{model_name}_epoch{epoch}", graph_dir, val=False)


            print(f"Epoch ({epoch + 1} / {max_epochs}). Training loss={epoch_loss:2.3f}, Training dice = {mean_dice_train:2.3f}, Evaluation dice metric={mean_dice_val:2.3f}", flush=True)

        # Save model every 50 epochs
        if (epoch % 100 == 0 and epoch != 0) or epoch + 1 == max_epochs:
            # save model
            checkpoint_path = os.path.join(model_dir, f'model_checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': epoch,
                'best_epoch': global_step_best,
                'epoch_loss_values': epoch_loss_values,
                'metric_values': metric_values,
                'metric_values_train': metric_values_train,
                'best_loss_values': epoch_loss_value_best,
                'best_metric_values': metric_value_best,
            }, checkpoint_path)
            print(f"Checkpoint saved! Epoch ({epoch + 1} / {max_epochs}).", flush=True)

        global_step = epoch


    # Finish Training
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training finished! Total time: {total_time:2.5f} seconds", flush=True) #CSV dataset: 104sec/epoch; #Cache Dataset rate=1: 50sec/epoch

    #debug
    # print("Spikes detected at:", spikes_epochs)

    return global_step, global_step_best, epoch_loss_values, metric_values, metric_values_train, metric_value_best, epoch_loss_value_best, ex_dices


# Call train(), need to re-Define first
epoch_loss_values = []
epoch_loss_value_best = 0.0 #DiceCELoss
metric_values_train =[]
metric_values = []
metric_value_best = 0.0 #DiceMetric
#debug
ex_dices = []

global_step_best = 0 #change to continue training
global_step = 0 #change to continue training 


# Call train() separately, for continuous training
num_example_train = len(train_dataset) #450
num_example_val = len(val_dataset) #113
max_epochs = max_epochs #500
#global_step += 1 #continue without repeat
global_step, global_step_best, epoch_loss_values, metric_values, metric_values_train, metric_value_best, epoch_loss_value_best, ex_dices = train(model, train_loader, loss_function, optimizer, scaler, device,
                                                                                           model_dir, bestval_save_dir,
                                                                                           metric_value_best, epoch_loss_value_best, global_step_best,
                                                                                           global_step, max_epochs, num_example_train, ex_dices)
# double check
print("Summary: ")
print(f"num train images: {num_example_train}, num val images: {num_example_val}")
print(f'learning rate: {lr}')
print(f"global_step: {global_step}, global_step_best: {global_step_best}")
print(f"epoch_loss_value_best: {epoch_loss_value_best}, metric_value_best: {metric_value_best}")
print("=============", flush=True)



"""#### Check training output (val)"""
print(f"Exporting loss & metrics graphs", flush=True)



# Check loss & metrics
plt.ioff()
plt.figure("train", (15, 6))
plt.subplot(1, 3, 1); plt.title("Epoch DiceCELoss"); plt.xlabel(f"Epochs ({num_example_train} images/epoch)")
x = [i+1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.plot(x, y)

plt.subplot(1, 3, 2); plt.title("Train DiceMetric"); plt.xlabel(f"Epochs ({num_example_train} images/10 epochs)")
x = [(i) for i in range(len(metric_values_train))] #temp: (i)*10
y = metric_values_train
plt.plot(x, y); plt.ylim(0, 1)

plt.subplot(1, 3, 3); plt.title("Val DiceMetric"); plt.xlabel(f"Epochs ({num_example_val} images/10 epochs)")
x = [(i) for i in range(len(metric_values))] #temp: (i)*10
y = metric_values
plt.plot(x, y); plt.ylim(0, 1)

plt.suptitle(f"Result from {model_name}")
plt.savefig(f'{model_dir}/train_val_loss_{model_name}.png')
plt.close()
