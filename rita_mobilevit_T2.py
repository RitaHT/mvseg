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
)

print("all packages imported", flush=True)

"""## Try Segmentation
### Put into dataset using MONAI
https://docs.monai.io/en/latest/transforms.html#dictionary-transforms

#### Directory
"""

#!nvcc --version
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}", flush=True)

# load 
print("Data loading...", flush=True)
cwd = os.getcwd() #/data/deasy/shiqin_data/intern24/
print(cwd, flush=True)
modality = "T2"
file_dir = './PROSTATEx/Files/slices_fcolor/'
csvFilePath_total = os.path.join(file_dir,f'slices_fcolor_total_{modality}.csv') #lesions_total.csv
csvFilePath_train = os.path.join(file_dir,f'overfit1_{modality}.csv') #slices_train_T2.csv
csvFilePath_val = os.path.join(file_dir,f'slices_val_{modality}.csv') #slices_val.csv

"""#### Load"""

train_transforms = Compose([
        LoadImaged(keys=["image", "label"],channel_dim=2), #default NibelReader #[384, 384, 3] #
        EnsureChannelFirstd(keys=["image", "label"]), #[3, 84, 128]
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")), #[3, 267, 383]~ [3, 407, 323]
        #ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0, b_max=1, clip=True), #T2??s
        NormalizeIntensityd(keys=["image"], channel_wise = True),

        #Resized(keys=["image", "label"], spatial_size=(384,384), anti_aliasing=True), #[3, 364, 364] #most T2 will be <384 after resampled 
        #CenterSpatialCropd(keys=["image", "label"], roi_size=(256,256)), #[3, 256, 256] 
        # testing different cropping effect
        CenterSpatialCropd(keys=["image", "label"], roi_size=(160,160)), #[3, 256, 256]
        Resized(keys=["image", "label"], spatial_size=(256,256), anti_aliasing=True), #[3, 288, 288]

        EnsureTyped(keys=["image", "label"], device=device, track_meta=False), #execute random transforms on GPU directly

        #data augmentations:
        RandRotated(keys=["image", "label"], prob=0.5, range_x=[-1.0, 1.0]), #random rotation
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)), #ColorJitter in PyTorch

        ToTensord(keys=["image", "label"]),
])

val_transforms = Compose( [
        LoadImaged(keys=["image", "label"],channel_dim=2), #default NibelReader #[84, 128, 3] #
        EnsureChannelFirstd(keys=["image", "label"]), #[3, 84, 128]
        Spacingd(keys=["image", "label"], pixdim=(0.5, 0.5, 3.0), mode=("bilinear", "nearest")), #[3, 267, 383]~ [3, 407, 323]
        #ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0, b_max=1, clip=True), #T2
        NormalizeIntensityd(keys=["image"], channel_wise = True),

        #Resized(keys=["image", "label"], spatial_size=(384,384), anti_aliasing=True), #[3, 364, 364]
        #CenterSpatialCropd(keys=["image", "label"], roi_size=(256,256)), #[3, 256, 256]
        # testing different cropping effect
        CenterSpatialCropd(keys=["image", "label"], roi_size=(160,160)), #[3, 256, 256]
        Resized(keys=["image", "label"], spatial_size=(256,256), anti_aliasing=True), #[3, 288, 288]

        EnsureTyped(keys=["image", "label"], device=device, track_meta=False), #execute random transforms on GPU directly
        # without data augmentations
        ToTensord(keys=["image", "label"]),
])

train_dataset_pre = CSVDataset( #change to CacheDataset for faster later:  train_dataset_pre
    src=csvFilePath_train,
    col_names=["ID",f"{modality}_image_slice", f"{modality}_mask_slice"],
    col_groups={"image": f"{modality}_image_slice", "label": f"{modality}_mask_slice"}, #create new col, easier later switch to T2
    transform=train_transforms,
)
val_dataset_pre = CSVDataset(
    src=csvFilePath_val,
    col_names=["ID","T2_image_slice", "T2_mask_slice"],
    col_groups={"image": "T2_image_slice", "label": "T2_mask_slice"}, #create new col, easier later switch to T2
    transform=val_transforms,
)

print(f"train_dataset_pre len: {len(train_dataset_pre)}", flush=True) #ADC: 450, T2: 454


train_dataset = CacheDataset(train_dataset_pre, cache_rate=1.0, progress=False)
val_dataset = CacheDataset(val_dataset_pre, cache_rate=1.0, progress=False)

print(f"dataset len: {len(train_dataset)}", flush=True)
print(f"dataset len: {len(val_dataset)}", flush=True)

print(f"dataset img: {train_dataset[0]['image'].shape}", flush=True)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)# num_workers=2, pin_memory=True)#, collate_fn=pad_list_data_collate) #(239) #still error?? #collate_fn=pad_list_data_collate
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)# num_workers=2, pin_memory=True) #(60)

print(len(train_loader))

i = 0
for batch_data in train_loader:
    i += 1
    if i == 1: #choose to see which example images
        break
input = batch_data["image"]
print(f"dataloader img: {input.shape}", flush=True) #[4,3,256,256]


print("Data loaded", flush=True)

"""### Try Training
Modify to MONAI / transformer trainer later
MONAI: https://docs.monai.io/en/stable/engines.html#trainer
HF(transformer): https://huggingface.co/docs/transformers/main/en/training#finetune-with-trainer
Reference training loop: https://colab.research.google.com/drive/1ebbMS5qNH0s7grA_cZSTu5MIo7vPXbYC?usp=sharing

#### Define
"""

# Define MobileViT_DeepLabV3 model class
class MobileViT_DeepLabV3(nn.Module):
    def __init__(self, backbone, seg_head):
        super(MobileViT_DeepLabV3, self).__init__()
        self.backbone = backbone
        self.seg_head = seg_head
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_intermediate=False):
        outputs_base = self.backbone(x)
        inputs_seg = outputs_base[-1].to(device) #[batch, 256/512/1024, 8/32, 8/32] feature maps #backprop???
        outputs_seg = self.seg_head(inputs_seg) #[batch, 3, 8/32, 8/32] MetaTensor
        # need softmax(multiclass)/sigmoid(binary) activation to turn feature map into segmentation probability
        outputs_active = self.sigmoid(outputs_seg) #[batch, 3, 32, 32] #ex.max=0.8551, min=0.1949
        outputs_upsampled = F.interpolate(outputs_active, size=x.shape[-2:], mode='bilinear', align_corners=False) #[batch, 3, 256, 256] #ex. max=0.8345, min=0.2594
        return outputs_upsampled 

# Confirm models
base_model = timm.create_model(
    'mobilevitv2_100.cvnets_in1k',
    pretrained=True,
    features_only=True,
    #output_stride=8,
).to(device)

seg_model = DeepLabHead(
    in_channels=512,  # Adjust if feature map from base_model differ
    num_classes=1,  # foreground/lesion, background
    #atrous_rates=(6,12,18) #from the paper, PyTorch default (12,24,36)
).to(device)

model = MobileViT_DeepLabV3(base_model, seg_model).to(device)

# Define Loss, Optimizer
torch.backends.cudnn.benchmark = True
#loss_function = BCEWithLogitsLoss()   #nn.CrossEntropyLoss()
loss_function = DiceCELoss(include_background=True, to_onehot_y=False, sigmoid=False, squared_pred=True) #1.0090 #to_onehot_y=True for [batch,1,3,256,256] #sigmoid already in the model? #why use squared_pred in denominator?
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.00005) #lr > weight_decay #as paper, base_model & seg_model?
scaler = torch.cuda.amp.GradScaler()
#scheduler?

# Define dice metric
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
discrete_pred = AsDiscrete(to_onehot=None, threshold=0.5)  #should be probability #to_onehot=number of classes, threshold=values
discrete_label = AsDiscrete(to_onehot=None, threshold=0.5) #originally not only [0,1] but PROSTATEx docu said it is binarized, bg=0 always

# evaluation loop
def validation(model, val_loader, dice_metric, device):
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            inputs_v, labels_v = (batch_data["image"].to(device), batch_data["label"].to(device),)# [batch, 3, 256, 256]
            with torch.cuda.amp.autocast():
                outputs_upsampled_v = model(inputs_v)
                outputs_upsampled_v = discrete_pred(outputs_upsampled_v) 
                labels_v = labels_v[:,1,:,:].unsqueeze(1) #[batch,1,256,256] #middle slice
                labels_v = discrete_label(labels_v)
            dice_metric(y_pred=outputs_upsampled_v, y=labels_v) #num_class=shape[1]
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    #print(f"Evaluation finished! Average loss = {avg_loss:2.5f}")
    model.train() #do i need this???
    return mean_dice_val

"""#### Train"""
# saving the model
cwd = os.getcwd() #/data/deasy/shiqin_data/intern24/
model_folder = "model_0812"
model_name = "T2_fcolor"
model_dir = os.path.join(cwd, f"{model_folder}/{model_name}") #/content/drive/MyDrive/harini_lab/model_X
if path.exists(model_dir) == False:
    os.makedirs(model_dir, exist_ok=True)
    print(f"Create new folder {model_dir}")
bestval_save_dir = os.path.join(model_dir, "model_bestval.pth")
print(f"model_dir: {model_dir}", flush=True)

# training loop
def train(model, train_loader, loss_function, optimizer, scaler, device, 
          model_dir, bestval_save_dir,
          metric_value_best, epoch_loss_value_best, global_step_best,
          global_step_start, max_epochs, num_example, resume=False):

    start_time = time.time()

    model.train()

    num_example = num_example
    global_step = global_step_start
    ori_resolution = (256, 256)

    print(f"Training starts! Set to run {max_epochs} epochs", flush=True)
    for epoch in range(global_step_start, max_epochs):
        epoch_loss = 0
        for batch_data in train_loader: 
            optimizer.zero_grad()
            inputs, labels = (batch_data["image"].to(device),batch_data["label"].to(device),)
            # Forward pass
            with torch.cuda.amp.autocast():
                outputs_upsampled = model(inputs) #[batch,1,256,256]
                labels = labels[:,1,:,:].unsqueeze(1) #[batch, 1,256,256] #middle slice
                loss = loss_function(outputs_upsampled, labels) 
            # Backward pass
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        epoch_loss /= num_example #average loss over this epoch
        epoch_loss_values.append(epoch_loss)

        if (epoch <= 40): #make sure the model is running
            print(f"Training {global_step + 1} epoch (loss={epoch_loss:2.5f})", flush=True)

        # Validation & Save Model
        if (epoch % 10 == 0 and epoch > 0) or epoch + 1 == max_epochs: #hardcoded to 10, 20 iter, may change later
            # Training accuracy
            mean_dice_train = validation(model, train_loader, dice_metric, device)
            metric_values_train.append(mean_dice_train)
            # Validation accuracy
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

            print(f"Epoch ({epoch + 1} / {max_epochs}). Training loss={epoch_loss:2.3f}, Training dice = {mean_dice_train:2.3f}, Evaluation dice metric={mean_dice_val:2.3f}", flush=True)

        # Save model every 50 epochs
        if (epoch % 100 == 0 and epoch != 0) or epoch + 1 == max_epochs:
            # save model
            checkpoint_path = os.path.join(model_dir, f'model_checkpoint_epoch_{epoch}.pth')
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
    
    print("Summary: ")
    print(f"num train images: {num_example_train}, num val images: {num_example_val}")
    print(f"global_step: {global_step}, global_step_best: {global_step_best}")
    print(f"epoch_loss_value_best: {epoch_loss_value_best}, metric_value_best: {metric_value_best}")
    print("=============", flush=True)
    return global_step, global_step_best, epoch_loss_values, metric_values, metric_values_train, metric_value_best, epoch_loss_value_best


# Call train(), need to re-Define first
epoch_loss_values = []
epoch_loss_value_best = 0.0
metric_values_train =[]
metric_values = []
metric_value_best = 0.0 #using DiceCELoss, change to dicemetric() later????

global_step_best = 0 #change to continue training
global_step = 0 #change to continue training 

# Call train() separately, for continuous training
num_example_train = len(train_dataset) #450
num_example_val = len(val_dataset) #113
max_epochs = 300 #500
#global_step += 1 #continue without repeat
global_step, global_step_best, epoch_loss_values, metric_values, metric_values_train, metric_value_best, epoch_loss_value_best = train(model, train_loader, loss_function, optimizer, scaler, device,
                                                                                           model_dir, bestval_save_dir,
                                                                                           metric_value_best, epoch_loss_value_best, global_step_best,
                                                                                           global_step, max_epochs, num_example_train)




"""#### Check training output (val)"""
print(f"Exporting loss & metrics graphs", flush=True)

# Check loss & metrics
plt.ioff()
plt.figure("train", (12, 6))
plt.subplot(1, 3, 1)
plt.title("Epoch DiceCELoss"); plt.xlabel(f"Epochs ({num_example_train} images/epoch)")
x = [i+1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.plot(x, y)

plt.subplot(1, 3, 2)
plt.title("Train DiceMetric"); plt.xlabel(f"Train Images Index ({num_example_train} images/10 epochs)")
x = [(i)*10 for i in range(len(metric_values_train))]
y = metric_values_train
plt.plot(x, y)
plt.ylim(0, 1)


plt.subplot(1, 3, 3)
plt.title("Val DiceMetric"); plt.xlabel(f"Validation Images Index ({num_example_val} images/10 epochs)")
x = [(i)*10 for i in range(len(metric_values))]
y = metric_values
plt.plot(x, y)
plt.ylim(0, 1)

plt.suptitle(f"Result from {model_name}")
plt.savefig(f'train_val_loss_{model_name}.png')
plt.close()


# Example inference on Training Set
print(f"Exporting example inference image on Training Set", flush=True)
model.eval()
val_i = 0
for batch_data in train_loader:
    val_i += 1
    if val_i == 1: #change
        break

inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device),) #[batch,3,256,256]
outputs_upsampled = model(inputs)
outputs_upsampled = outputs_upsampled.detach() #grad
outputs_upsampled_v = discrete_pred(outputs_upsampled)
print(outputs_upsampled_v.shape)
labels_v = labels[:,1,:,:].unsqueeze(1) #[batch,1,256,256]
labels_v = discrete_label(labels_v)
print(labels_v.shape)

i = 0 #change if needed
dice_metric(y_pred=outputs_upsampled_v[i,:,:,:], y=labels_v[i,:,:,:]) #compute dice for one image in this batch #num_class=shape[1]
example_dice_train = dice_metric.aggregate().item() # 0.004
dice_metric.reset()
print(f"example_dice_train: {example_dice_train}")

results_upsampled = outputs_upsampled.cpu().detach().numpy()
inputs_for_eval = inputs.cpu()
labels_for_eval = labels.cpu()

threshold = 0.5
binary_masks = (results_upsampled > threshold) #[batch,1,256,256]

fig = plt.figure(figsize=(5, 15))
for j in range(3):
    a = fig.add_subplot(12, 3, j+1); a.set_title("original image", fontsize=8); a.axis("off");
    plt.imshow(inputs_for_eval[0,j,:,:].T, cmap="gray")

    a = fig.add_subplot(12, 3, j+1+3); a.set_title("original mask", fontsize=8);a.axis("off");
    plt.imshow(labels_for_eval[0,j,:,:].T, cmap="gray")

a = fig.add_subplot(12, 3, 8);a.set_title("Interp output", fontsize=8); a.axis("off");
imgplot = plt.imshow(results_upsampled[0,:,:,:].T,vmin=0,vmax=1)

a = fig.add_subplot(12, 3, 11);a.set_title("Interp output(mask)", fontsize=8);a.axis("off");
plt.imshow(binary_masks[0,:,:,:].T)

fig.suptitle("Example slices from val_dataset {}".format(batch_data["ID"][0]),fontsize=10)
cbar_ax = fig.add_axes([0.93, 0.64, 0.01, 0.2]) #[right,up,thickness,length]
fig.colorbar(imgplot, cax=cbar_ax)

plt.savefig(f'example_train_{model_name}.png')
plt.close()

# Example inference on Training
print(f"Exporting example inference image on Validation Set", flush=True)

model.eval()
val_i = 0
for batch_data in val_loader: #batchsize=4
    val_i += 1
    if val_i == 6: #change #ProstateX-0169-Finding1_slices_0
        break
inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device),)
outputs_upsampled = model(inputs)
outputs_upsampled_v = discrete_pred(outputs_upsampled)
print(outputs_upsampled_v.shape)
labels_v = labels[:,1,:,:].unsqueeze(1) #[batch,1,256,256]
labels_v = discrete_label(labels_v)
print(labels_v.shape)

dice_metric(y_pred=outputs_upsampled_v, y=labels_v) #num_class=shape[1]
mean_dice_val = dice_metric.aggregate().item() # 0.004
dice_metric.reset()
print(f"mean_dice_val: {mean_dice_val}")


results_upsampled = outputs_upsampled.cpu().detach().numpy()
inputs_for_eval = inputs.cpu()
labels_for_eval = labels.cpu()

threshold = 0.5
binary_masks = (results_upsampled > threshold) #[batch,1,256,256]

fig = plt.figure(figsize=(5, 15))
for j in range(3):
    a = fig.add_subplot(12, 3, j+1); a.set_title("original image", fontsize=8); a.axis("off");
    plt.imshow(inputs_for_eval[0,j,:,:].T, cmap="gray")

    a = fig.add_subplot(12, 3, j+1+3); a.set_title("original mask", fontsize=8);a.axis("off");
    plt.imshow(labels_for_eval[0,j,:,:].T, cmap="gray")

a = fig.add_subplot(12, 3, 8);a.set_title("Interp output", fontsize=8); a.axis("off");
imgplot = plt.imshow(results_upsampled[0,:,:,:].T,vmin=0,vmax=1)

a = fig.add_subplot(12, 3, 11);a.set_title("Interp output(mask)", fontsize=8);a.axis("off");
plt.imshow(binary_masks[0,:,:,:].T)

fig.suptitle("Example slices from val_dataset {}".format(batch_data["ID"][0]),fontsize=10)
cbar_ax = fig.add_axes([0.93, 0.64, 0.01, 0.2]) #[right,up,thickness,length]
fig.colorbar(imgplot, cax=cbar_ax)

plt.savefig(f'example_val_{model_name}.png')
plt.close()