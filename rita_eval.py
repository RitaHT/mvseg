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
modality = "ADC"
file_dir = './PROSTATEx/Files/slices_repeated/'
csvFilePath_total = os.path.join(file_dir,'slices_repeated_total.csv') #lesions_total.csv
csvFilePath_train = os.path.join(file_dir,f'slices_train_{modality}.csv')#slices_train_ADC.csv
csvFilePath_val = os.path.join(file_dir,f'slices_val_{modality}.csv') #slices_val_ADC.csv

"""#### Load"""

train_transforms = Compose([
        LoadImaged(keys=["image", "label"],channel_dim=2), #default NibelReader #[84, 128, 3] #
        EnsureChannelFirstd(keys=["image", "label"]), #[3, 84, 128]
        Spacingd(keys=["image", "label"], pixdim=(3.0, 0.5, 0.5), mode=("bilinear", "nearest")), #0.5 T2 res: [3, 333, 506] #0.625: [3, 267, 383]~ [3, 407, 323]
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3500, b_min=0, b_max=1, clip=True), #ADC
        #NormalizeIntensityd(keys=["image"], channel_wise = True),

        #Resized(keys=["image", "label"], spatial_size=(384,384), anti_aliasing=True), #[3, 364, 364]
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
        Spacingd(keys=["image", "label"], pixdim=(3.0, 0.5, 0.5), mode=("bilinear", "nearest")), #0.5 T2 res: [3, 333, 506] #0.625: [3, 267, 383]~ [3, 407, 323]
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3500, b_min=0, b_max=1, clip=True), #ADC
        #NormalizeIntensityd(keys=["image"], channel_wise = True),

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
    col_names=["chID",f"{modality}_image_slice", f"{modality}_mask_slice"],
    col_groups={"image": f"{modality}_image_slice", "label": f"{modality}_mask_slice",
                "ID": "chID",
                "image_path": f"{modality}_image_slice",
                "label_path": f"{modality}_mask_slice"}, #create new col, easier later switch to T2
    transform=train_transforms,
)
print(f"train_dataset_pre len: {len(train_dataset_pre)}", flush=True)

val_dataset_pre = CSVDataset(
    src=csvFilePath_val,
    col_names=["chID",f"{modality}_image_slice", f"{modality}_mask_slice"],
    col_groups={"image": f"{modality}_image_slice", "label": f"{modality}_mask_slice",
                "ID": "chID",
                "image_path": f"{modality}_image_slice",
                "label_path": f"{modality}_mask_slice"}, #create new col, easier later switch to T2
    transform=val_transforms,
)

print(f"val_dataset_pre len: {len(val_dataset_pre)}", flush=True)


train_dataset = CacheDataset(train_dataset_pre, cache_rate=1.0, progress=False)
val_dataset = CacheDataset(val_dataset_pre, cache_rate=1.0, progress=False)

print(f"dataset len: {len(train_dataset)}", flush=True)
print(f"dataset len: {len(val_dataset)}", flush=True)

print(f"dataset img: {train_dataset[0]['image'].shape}", flush=True)

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)# num_workers=2, pin_memory=True)#, collate_fn=pad_list_data_collate) #(239) #still error?? #collate_fn=pad_list_data_collate
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)# num_workers=2, pin_memory=True) #(60)

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

# Call train(), need to re-Define first
epoch_loss_values = []
epoch_loss_value_best = 0.0
metric_values_train =[]
metric_values = []
metric_value_best = 0.0 #using DiceCELoss, change to dicemetric() later????

global_step_best = 0 #change to continue training
global_step = 0 #change to continue training 

num_example_train = len(train_dataset) #450
num_example_val = len(val_dataset) #113

# optional: load and resume training
resume_model_folder = "model_0812"
resume_model_name = "ADC_repeated_combined"
resume_model_dir = os.path.join(cwd, f"{resume_model_folder}/{resume_model_name}/model_checkpoint_epoch_299.pth") # model_checkpoint_epoch_500.pth")
print(f"resuming from {resume_model_name}")
checkpoint = torch.load(resume_model_dir)

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
global_step = checkpoint['epoch']
global_step_best = checkpoint['best_epoch']
epoch_loss_values = checkpoint['epoch_loss_values']
epoch_loss_value_best = checkpoint['best_loss_values']
metric_values_train = checkpoint['metric_values_train']
metric_values = checkpoint['metric_values']
metric_value_best = checkpoint['best_metric_values']

print(f"num train images: {num_example_train}, num val images: {num_example_val}")
print(f"global_step: {global_step}, global_step_best: {global_step_best}")
print(f"epoch_loss_value_best: {epoch_loss_value_best}, metric_value_best: {metric_value_best}")
print(len(epoch_loss_values))
print(len(metric_values))


# evaluate & saving image
resume_model_folder = resume_model_folder #"model_0808"
resume_model_name = resume_model_name     #"ADC_check_dice"
image_output_dir = os.path.join(cwd, f"{resume_model_folder}/{resume_model_name}/val_output") # model_checkpoint_epoch_500.pth")
os.makedirs(image_output_dir, exist_ok=True)
print(f"image_output_dir: {image_output_dir}")


dice_2D = []
IDs = []
image_paths = []
output_paths = []
batch = 0
for batch_data in val_loader:
    batch += 1
    if batch > 1:
        break
    inputs, labels, ID, image_path = (batch_data["image"].to(device), batch_data["label"].to(device), 
                          batch_data['ID'], batch_data["image_path"])# [batch, 3, 256, 256]
    outputs_upsampled = model(inputs)
    outputs_upsampled_v = discrete_pred(outputs_upsampled) 
    #print(f"output shape: {outputs_upsampled_v.shape}") #[batch,1,256,256], just one slice output
    labels_v = labels[:,1,:,:].unsqueeze(1) 
    labels_v = discrete_label(labels_v)
    #print(f"label  shape: {labels_v.shape}") #[batch,1,256,256] #middle slice

    inputs_for_eval = inputs.cpu()
    labels_for_eval = labels.cpu()
    threshold = 0.5
    results_upsampled = outputs_upsampled.cpu().detach().numpy()
    result_masks = (results_upsampled > threshold) #[batch,1,256,256]

    for i in range(inputs_for_eval.shape[0]): #batch size
        dice_metric(y_pred=outputs_upsampled_v[i,:,:,:], y=labels_v[i,:,:,:]) #each image within batch #num_class=shape[1]
        example_dice_val = dice_metric.aggregate().item() # 0.004
        dice_2D.append(example_dice_val)
        IDs.append(ID[i])
        image_paths.append(image_path[i])
        dice_metric.reset()
        print(f"example_dice_val: {example_dice_val}")

        
        fig = plt.figure(figsize=(5, 15))
        for j in range(3):
            a = fig.add_subplot(12, 3, j+1); a.set_title("original image", fontsize=8); a.axis("off");
            plt.imshow(inputs_for_eval[i,j,:,:].T, cmap="gray")

            a = fig.add_subplot(12, 3, j+1+3); a.set_title("original mask", fontsize=8);a.axis("off");
            plt.imshow(labels_for_eval[i,j,:,:].T, cmap="gray")

        #a = fig.add_subplot(12, 3, 8);a.set_title("Interp output", fontsize=8); a.axis("off");
        #imgplot = plt.imshow(results_upsampled[0,:,:,:].T,vmin=0,vmax=1)

        a = fig.add_subplot(12, 3, 8);a.set_title("output mask", fontsize=8);a.axis("off");
        plt.imshow(result_masks[i,:,:,:].T)

        a = fig.add_subplot(12, 3, 11); a.set_title("Overlay", fontsize=8); a.axis("off")
        plt.imshow(inputs_for_eval[i,1,:,:].T, cmap="gray") #middle slice
        plt.contour(labels_for_eval[i,1,:,:].T,  colors='red', linewidths = 0.6)
        plt.contour(result_masks[i,0,:,:].T,  colors='lime', linewidths = 0.8) #[batch,1,256,256]


        fig.suptitle(f"val_dataset {ID[i]} \n example_dice_coeff = {example_dice_val}",fontsize=10)
        #cbar_ax = fig.add_axes([0.93, 0.64, 0.01, 0.2]) #[right,up,thickness,length]
        #fig.colorbar(imgplot, cax=cbar_ax)

        output_filename = f"{os.path.basename(image_path[i]).replace('.nii.gz', '')}_seg.png"
        output_path = f'{image_output_dir}/{output_filename}'
        plt.savefig(output_path)
        plt.close()
        output_paths.append(output_path)
        print(f"saved {output_path}")

# saving individual dice to csv
dice_2D_avg = np.average(dice_2D)
print(f"dice_2D_avg: {dice_2D_avg}")
#dice_metric.reset()

df = pd.DataFrame({
    'ID': IDs,
    'dice_coeff': dice_2D,
    'input_image_path': image_paths,
    'output_image_path': output_paths,
})
df.to_csv(f"{cwd}/{resume_model_folder}/{resume_model_name}/dice_val.csv", index=False)

# combined with previous info
prev_info_df = csvFile = pd.read_csv(os.path.join(cwd,'PROSTATEx/Files/slices_only/slices_info_ADC.csv'))
merged_df = pd.merge(df, prev_info_df, on='ID', how='left')

merged_df.to_csv(f"{cwd}/{resume_model_folder}/{resume_model_name}/dice_val_merged.csv", index=False)

print(f"Evaluation finished! CSV saved! dice_2D_avg: {dice_2D_avg}")
