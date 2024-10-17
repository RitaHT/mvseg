
print("Start running", flush=True) 

import os
from os import listdir, path
import sys
import time
import numpy as np
import nibabel as nib #for reading .nii.gz format MRI files
import matplotlib.pyplot as plt #use pyplot in matplotlib instead, pylab is about to archived
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import monai
from monai.data import CacheDataset, DataLoader, decollate_batch, load_decathlon_datalist
from monai.data.utils import pad_list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from torch.nn import BCEWithLogitsLoss
from monai.metrics import DiceMetric
from monai.transforms.transform import Transform, MapTransform
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscreted, AsDiscrete, 
    Compose,
    LoadImaged, 
    Resized,
    Orientationd,
    SpatialCropd,
    CenterSpatialCropd,
    RandSpatialCropd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged, 
    ScaleIntensityRangePercentilesd,
    Spacingd,
    SpatialPadd,
    EnsureTyped,
    ToTensord,
    RandRotated, RandRotate90d,
    RandFlipd,
    RandAdjustContrastd,
    RandGaussianNoised,
    Rand2DElasticd,
    RandShiftIntensityd,
)


import argparse #args
parser = argparse.ArgumentParser(description='segmentation pipeline')
parser.add_argument('--distributed', action='store_true', help='start distributed training') #if the --distributed flag is passed when running the script, the argument will be set to True. If the flag is not provided, the argument will default to False.
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
parser.add_argument('--crop_margin', default=60, type=int, help='for CropForegroundd') #for crop
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction') #128*128*128
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction') #64,64,64
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--a_min_ct', default=-115, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max_ct', default=235, type=float, help='a_max in ScaleIntensityRanged')  #Jue: (-750,1750) only do to CT, not to PET
#For CT: soft tissue window 350-400, window level 20-60
#HU (-155,195) window 350, 20, (-115, 235) window 350, 60
parser.add_argument('--a_min_pet', default=0, type=float, help='a_min in ScaleIntensityRanged') #SUV #@prostate clipping???
parser.add_argument('--a_max_pet', default=0.2, type=float, help='a_max in ScaleIntensityRanged') 
parser.add_argument('--lower_pet', default=5, type=float, help='a_max in ScaleIntensityRanged')  #SUV 
parser.add_argument('--upper_pet', default=95, type=float, help='a_min in ScaleIntensityRanged') #ex. CHUP-075: (0,1) but ITK auto contrast adjust to (0,0.07) with window 1st,99th percentile
#For PET: SUV = radioactivity within X time/[injection dose (MBq)/patient's weight (kg)] 
#For the extreme small ones:  raw PET count? without calibration factor? decay-corrected activity per voxel??? not yet SUV?
#HECKTOR said some patient weight not available and assume 75kg
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=1, type=int, help='number of output channels') #0: background, 1: cancer, 2: lymph node #HECKTOR
parser.add_argument('--batch_size', default=4, type=int, help='number of batch size')
parser.add_argument('--max_epochs', default=300, type=int, help='max number of training epochs') #1 #50
parser.add_argument('--min_epochs', default=0, type=int, help='min number of training epochs') #0 #for resume training
#need to update the resume train
parser.add_argument('--modality', default='PET', type=str, help='modality (PET or CT)')
parser.add_argument('--k_folds', default=3, type=int, help='folds for cross validation')
parser.add_argument('--test_size', default=0.25, type=float, help='train/test split ratio') #0.2
parser.add_argument('--cache_rate', default=0.3, type=float, help='CacheDataset cache_rate') #0.2
parser.add_argument('--cache_num', default=40, type=int, help='CacheDataset cache_num') #10
parser.add_argument('--patience', default=10, type=int, help='early stopping patience') #10
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate') #1e-4
parser.add_argument('--weight_decay', default=5e-5, type=float, help='weight decay') #1e-5
#plotting:
parser.add_argument('--val_i_stop', default=6, type=int, help='number of validation images to stop') #6
#directory:
parser.add_argument('--model_folder', default='model_1016', type=str, help='model directory') #model_1010_scalepercentile
parser.add_argument('--model_name', default='unet', type=str, help='model name') #PET_unet
parser.add_argument('--graph_dir', default='graphs', type=str, help='graph directory') #graphs
parser.add_argument('--file_dir', default='./', type=str, help='file directory') #/lila/data/deasy/shiqin_data/hecktor_unet/
parser.add_argument('--json_file', default=f'train.json', type=str, help='json file') #ex_data_CT.json
parser.add_argument('--json_data_list_key', default='train_data', type=str, help='json file') #ex_data_CT.json

""" Classes: """
class MergeClassesd(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            label = data[key]
            label = torch.where(label == 2, 1, label)  # Combine label 2 into label 1
            data[key] = label
        return data

""" Functions:"""
def validation(model, loader, dice_metric, post_pred, post_label):
    with torch.no_grad():
        mean_dice = 0.0
        for val_data in loader:

            val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            if val_inputs.shape != val_labels.shape:
                print(f"val_inputs shape: {val_inputs.shape}", flush=True)
                print(f"val_labels shape: {val_labels.shape}", flush=True)
            with torch.cuda.amp.autocast():
                val_outputs = model(val_inputs)
            val_outputs_list = decollate_batch(val_outputs) #length = 4, each is [1,128,128,128]
            val_outputs_convert = [post_pred(val_output_tensor) for val_output_tensor in val_outputs_list] #length = 4, each is [3,128,128,128]
            # val_outputs = post_pred(val_outputs) #[batch,3,128,128,128]
            val_labels_list = decollate_batch(val_labels) #length = 4, each is [1,128,128,128]
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list] #length = 4, each is [3,128,128,128]
            # val_labels = post_label(val_labels) #[batch,3,128,128,128] #after to_onehot=out_channels
            # because post_label=AsDiscrete doesn't seem to work for []
            if val_outputs_convert[0].shape != val_labels_convert[0].shape:
                print(f"len(val_outputs_convert): {len(val_outputs_convert)}", flush=True)
                print(f"val_outputs_convert[0] shape: {val_outputs_convert[0].shape}", flush=True)
                print(f"len(val_labels_convert): {len(val_labels_convert)}", flush=True)
                print(f"val_labels_convert[0] shape: {val_labels_convert[0].shape}", flush=True)

            dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return mean_dice

def export_ex_seg(model, batch_data, model_name, graph_dir, 
                  dice_metric, post_pred, post_label, #args,
                  i = 0, slice_num = 63,val=True,save_nifti=False):
    with torch.no_grad():
        inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device)) #
        IDs = inputs.meta['filename_or_obj'] #['imagesTr/CHUM-008__CT.nii.gz', 'imagesTr/CHUP-075__CT.nii.gz', 'imagesTr/CHUM-010__CT.nii.gz', 'imagesTr/CHUM-013__CT.nii.gz']
        affines = inputs.meta['affine']
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
    outputs = outputs.detach() #grad
    val_outputs_list = decollate_batch(outputs) #length = 4, each is [1,128,128,128]
    val_outputs_convert = [post_pred(val_output_tensor) for val_output_tensor in val_outputs_list] #length = 4, each is [3,128,128,128]
    val_labels_list = decollate_batch(labels) #length = 4, each is [1,128,128,128]
    val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list] #length = 4, each is [3,128,128,128]

    if val_outputs_convert[0].shape != (1, 128, 128, 128): #do not print, when normal
        print(f"output[0] shape: {val_outputs_convert[0].shape}", flush=True)
    if val_labels_convert[0].shape  != (1, 128, 128, 128): #do not print, when normal
        print(f"label[0]  shape: {val_labels_convert[0].shape}", flush=True)

    #dice on just 1 image, not the whole batch
    dice_metric(y_pred=val_outputs_convert[i], y=val_labels_convert[i]) #compute dice for one image in this batch #num_class=shape[1] #must be [i,:,...]=[batch,channel,spatial]
    example_dice= dice_metric.aggregate().item() # 0.004
    dice_metric.reset()
    print(f"example_dice: {example_dice}")

    
    # Save the outputs as NIfTI files #not using
    if save_nifti:
        for save_i, output in enumerate(val_outputs_convert):
            # Convert the tensor to a NumPy array
            output_np = output.cpu().numpy()  # Make sure to move to CPU
            # Reshape the output to (128, 128, 128) 
            output_np = (output_np > 0.5).astype(np.int16)   # threshold for binary, mask should make no difference converting to int16     # argmax: multi-class: (3, 128, 128, 128)
            
            # Create a NIfTI image
            nifti_image = nib.Nifti1Image(output_np, affines[save_i])  # error: int64 not compatible with nibabel
            # Save the NIfTI image
            output_filename = f"{IDs[save_i].split('/')[-1].replace('.nii.gz', '_output.nii.gz')}"
            nib.save(nifti_image, f"{graph_dir}/{output_filename}")
            print(f"Saved output to {output_filename}")

    inputs = inputs.detach() #[batch, 1, 128,128,128]
    val_inputs_list = decollate_batch(inputs) #4* [1,128,128,128]
    input_tensor = val_inputs_list[i].cpu().numpy() # [1,128,128,128]
    label_tensor = val_labels_convert[i].cpu().numpy() # [1,128,128,128]
    output_tensor = val_outputs_convert[i].cpu().numpy() # [1,128,128,128] #already AsDiscrete
    # #debug
    # # print(f"input_tensor shape: {input_tensor.shape}")
    # # print(f"input_tensor min: {input_tensor.min()}, max: {input_tensor.max()}", flush=True) 
    # #train: input_tensor min: 0.0, max: 1.8000091586145572e-05
    # #val: input_tensor min: 7.245863908877936e-10, max: 5.0988655857509e-05

    fig = plt.figure(figsize=(5, 5))
    #assume 0: background, 1: cancer & lymph node
    a = fig.add_subplot(2, 2, 1); a.set_title("image", fontsize=8); a.axis("off")
    plt.imshow(np.rot90(input_tensor[0,:,:,slice_num]), cmap="gray")
    a = fig.add_subplot(2, 2, 2); a.set_title("label(cancer & lymph)", fontsize=8);a.axis("off")
    plt.imshow(np.rot90(label_tensor[0,:,:,slice_num]), cmap="gray")
    a = fig.add_subplot(2, 2, 3);a.set_title("Output(cancer & lymph)", fontsize=8);a.axis("off")
    plt.imshow(np.rot90(output_tensor[0,:,:,slice_num]))
    a = fig.add_subplot(2, 2, 4); a.set_title("Overlay(cancer & lymph)", fontsize=8); a.axis("off")
    plt.imshow(np.rot90( input_tensor[0,:,:,slice_num]), cmap="gray") #middle slice
    plt.contour(np.rot90(label_tensor[0,:,:,slice_num]),  colors='red', linewidths = 0.4)
    plt.contour(np.rot90(output_tensor[0,:,:,slice_num]),  colors='lime', linewidths = 0.4) #[batch, 1, 256,256] output #result must be 2D, use [i,0,..] not [i,:,..]

    if val:
        fig.suptitle(f"val_dataset {IDs[i][9:]} \n example_dice_coeff = {example_dice:.4f}",fontsize=10) #imagesTr/CHUM-008__CT.nii.gz
        plt.savefig(f'{graph_dir}/ex_val_{model_name}.png')
    else:
        fig.suptitle(f"train_dataset {IDs[i][9:]} \n example_dice_coeff = {example_dice:.4f}",fontsize=10)
        plt.savefig(f'{graph_dir}/ex_train_{model_name}.png')
    plt.close()
    # print(f"Exported example image", flush=True)
    return example_dice


def export_ex_seg_train_val(model, model_name, graph_dir, 
                            train_loader, val_loader, epoch,
                            dice_metric, post_pred, post_label,
                            train_i_stop = 0, val_i_stop = 6, i = 0, slice_num = 64, save_nifti=False):
    print(f"Exporting eximage on Training Set", flush=True)
    train_i = 0
    for batch_data in train_loader:
        train_i += 1
        if train_i == train_i_stop: #change if needed
            break
    export_ex_seg(model, batch_data, f"{model_name}_epoch{epoch}", graph_dir, dice_metric, post_pred, post_label, slice_num=slice_num, val=False, save_nifti=save_nifti)
    
    print(f"Exporting eximage on Validation Set", flush=True)
    val_i = 0
    for batch_data in val_loader: #
        val_i += 1
        if val_i == val_i_stop: #change if needed#ProstateX-0169-Finding1_slices_0
            break
    export_ex_seg(model, batch_data, f"{model_name}_epoch{epoch}", graph_dir, dice_metric, post_pred, post_label, slice_num=slice_num, val=True, save_nifti=save_nifti)


def export_ex(val_data_example, modality, num_slice=12, val=True):        
    plt.ioff()
    fig = plt.figure("image", (15, 7))
    input = val_data_example["image"]
    label = val_data_example["label"]
    print(f"image shape: {input.shape}")
    print(f"label shape: {label.shape}")
    ID = input.meta["filename_or_obj"]
    shape = input.shape[-1]
    # print(shape)
    
    for i in range(num_slice):
        slice = i*5+(shape//4)
        if slice > shape:
            break
        a = fig.add_subplot(3, num_slice, i + 1); a.set_title(f"image slice {slice}", fontsize=6); a.axis("off")
        plt.imshow(np.rot90(input[0, :, :, slice].detach().cpu()), cmap="gray")
        a = fig.add_subplot(3, num_slice, i + num_slice + 1); a.set_title(f"label slice {slice}", fontsize=6); a.axis("off")
        plt.imshow(np.rot90(label[0, :, :, slice].detach().cpu()))
        a = fig.add_subplot(3, num_slice, i + num_slice*2 + 1); a.set_title(f"overlay slice {slice}", fontsize=6); a.axis("off")
        plt.imshow(np.rot90(input[0, :, :, slice].detach().cpu()), cmap="gray")
        plt.contour(np.rot90(label[0, :, :, slice].detach().cpu()), colors='red', linewidths = 0.6)

    if val:
        fig.suptitle(f"Validation_{modality}: {ID}")
        plt.savefig('eximage_val.png')
    else:
        fig.suptitle(f"Training_{modality}: {ID}")
        plt.savefig('eximage_train.png')
    plt.close()


# Define the transforms for training and validation sets
def get_transforms(args):
    if args.modality == "CT":
        args.a_min = args.a_min_ct
        args.a_max = args.a_max_ct
        # args.upper = args.upper_ct #do CT need this? isn't it absolute? 
        # args.lower = args.lower_ct
    elif args.modality == "PET":
        args.a_min = args.a_min_pet
        args.a_max = args.a_max_pet
        args.upper = args.upper_pet
        args.lower = args.lower_pet
    else:
        raise ValueError("modality must be either CT or PET")
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        MergeClassesd(keys=["label"]),
        ScaleIntensityRanged(keys=["image"], 
                                a_min=args.a_min, a_max=args.a_max, 
                                b_min=args.b_min, b_max=args.b_max, clip=True),
        Spacingd(keys=["image", "label"], 
                    pixdim=(args.space_x, args.space_y, args.space_z), 
                    mode=("bilinear", "nearest"), ),
        CropForegroundd(keys=["image", "label"], 
                        source_key="label", 
                        allow_smaller=True, #if false, will pad automatically
                        margin=(args.crop_margin,args.crop_margin,args.crop_margin)), #
        SpatialPadd(keys=["image", "label"], 
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
        RandSpatialCropd(keys=["image", "label"], 
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                random_size=False),
        EnsureTyped(keys=["image", "label"], device=device),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.5, 2.0)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), #when using 3D UNet        
        RandShiftIntensityd(keys=["image"], prob=args.RandShiftIntensityd_prob, offsets=0.2),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        MergeClassesd(keys=["label"]),
        ScaleIntensityRanged(keys=["image"], 
                                a_min=args.a_min, a_max=args.a_max, 
                                b_min=args.b_min, b_max=args.b_max, clip=True),
        Spacingd(keys=["image", "label"], 
                    pixdim=(args.space_x, args.space_y, args.space_z), 
                    mode=("bilinear", "nearest"), ),
        CropForegroundd(keys=["image", "label"], 
                        source_key="label", 
                        allow_smaller=False, #if false, will pad automatically?
                        margin=(args.crop_margin,args.crop_margin,args.crop_margin)), #hardcoded
        SpatialPadd(keys=["image", "label"], 
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z)), 
        CenterSpatialCropd(keys=["image", "label"],  #ensure the same size for all images
                            roi_size=(args.roi_x, args.roi_y, args.roi_z)), #do I need this??? 
        EnsureTyped(keys=["image", "label"], device=device), #track_meta=False
        ToTensord(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms



""" Directory & Constants: """
#!nvcc --version
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}", flush=True)

# 
def main():
    args = parser.parse_args() 
    main_worker(gpu=0, args=args)

def main_worker(gpu, args):
    mp.set_start_method('spawn')
    print("not distributed training")
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)

    # Defining model name
    cwd = os.getcwd() #/data/deasy/shiqin_data/intern24/s
    # model_folder = "model_1016" #model_1010_scalepercentile
    # model_name = f"{args.modality}_unet"
    # model_dir = os.path.join(cwd, f"{model_folder}/{model_name}") #
    # graph_dir = os.path.join(model_dir, "graphs")
    model_name = f"{args.modality}_{args.model_name}"
    model_dir = os.path.join(cwd, f"{args.model_folder}/{model_name}") #
    graph_dir = os.path.join(model_dir, args.graph_dir) #graphs
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
    # bestval_save_dir = os.path.join(model_dir, "model_bestval.pth")
    print(f"model_dir: {model_dir}", flush=True)

    # Load data using the MONAI's decathlon format reader, specifically for Hecktor dataset (ct, pet, label)
    file_dir = './'
    json_file =  f"{file_dir}/train_json_{args.modality}_preprocessed.json" #ex_data_CT.json
    datalist = load_decathlon_datalist(json_file, is_segmentation=True, 
                                    data_list_key="train_data", base_dir="./")
    json_file =  f"{args.file_dir}/{args.json_file}"#train_json_{args.modality}_preprocessed.json" #ex_data_CT.json
    datalist = load_decathlon_datalist(json_file, is_segmentation=True, 
                                    data_list_key=args.json_data_list_key, base_dir=args.file_dir) #train_data, val_data, test_data
    print("length of datalist: ", len(datalist), flush=True)

    """ Model: """
    # Define UNet model 
    class UNet_Segmenter(nn.Module):
        def __init__(self, unet):
            super(UNet_Segmenter, self).__init__()
            self.unet = unet
            # self.sigmoid = nn.Sigmoid() #binary segmentation
            # self.softmax = nn.Softmax(dim=1) #multi-class segmentation

        def forward(self, x):
            outputs_seg = self.unet(x)  # [batch, 1, 128, 128, 128] if using 3D UNet, with out_channels=3
            # DiceCELoss use BCEwithLogits, should output logits directly
            # outputs_activate = self.sigmoid(outputs_seg) #binary segmentation 
            # outputs_activate = self.Softmax(outputs_seg) #multi-class segmentation
            return outputs_seg #outputs_activate 
    # initialize the model in the fold loop

    # Define Loss, Optimizer
    torch.backends.cudnn.benchmark = True
    # lr = 0.0005
    # weight_decay=0.00005
    loss_function = DiceCELoss(#include_background=False,  #False #try change to True to see why cannot differentiate cancer & lymph
                                # to_onehot_y=True,  #number of classes inferred from `input` (``input.shape[1]``)=3 from model definition #handle the not one-hot label: but if those with just tumor?
                                sigmoid=True, #use in DiceLoss only, BCEwithLogitsLoss has its own
                                softmax=False, 
                                squared_pred=True,
                                smooth_nr=0.0, smooth_dr=1e-6) #1.0090 #to_onehot_y=True for [batch,1,3,256,256] #sigmoid already in the model? #why use squared_pred in denominator?

    # Define dice metric
    dice_metric = DiceMetric(#include_background=False, 
                             reduction="mean", #or mean_channel?
                             get_not_nans=False,
                             ignore_empty=True, #default Ture: NaN value will be set for empty ground truth cases.
                             num_classes = None) #if None, y_pred.shape[1] will be used #num_classes=1 
    post_pred =  Compose([Activations(sigmoid=True), 
                          AsDiscrete(to_onehot=None, #args.out_channels, 
                                     threshold=0.5)])
                                     #argmax=True)  #should be probability #to_onehot=number of classes, threshold=values
    post_label = AsDiscrete(to_onehot=None, threshold=0.5)

    """ Training: """
    # Split data using KFold from sklearn
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42) #fix random state so that resume trainin on same data, or need to save separate json
    #check time
    start_time = time.time()

    # Main training loop over K-folds
    for fold, (train_idx, val_idx) in enumerate(kf.split(datalist)):
        print(f"Training fold {fold+1}/{args.k_folds}, set to run {args.max_epochs} epochs", flush=True)

        #debug
        start_time_epoch = time.time()
        # Split the dataset
        train_files = [datalist[i] for i in train_idx]
        val_files = [datalist[i] for i in val_idx]
        print("length of train_files: ", len(train_files), flush=True)
        print("length of val_files: ", len(val_files), flush=True)

        train_transforms, val_transforms = get_transforms(args)

        print("Data loading...", flush=True)
        train_dataset = CacheDataset(data=train_files, transform=train_transforms, 
                                    num_workers=4,
                                    cache_rate=args.cache_rate,
                                    cache_num=args.cache_num)#, progress=False) #cache_num will take min
        print(f"train dataset len: {len(train_dataset)}", flush=True) #335 #debug: 
        val_dataset = CacheDataset(data=val_files, transform=val_transforms, 
                                num_workers=4,
                                cache_rate=args.cache_rate,
                                cache_num=args.cache_num)#, progress=False)
        print(f"val dataset len  : {len(val_dataset)}", flush=True) #84 #debug: 

        if fold == 0:
            # example images
            print("Exporting val example images...", flush=True)
            val_data_example = val_dataset[0] #a dictionary: image,foreground_start_coord, foreground_end_coord
            export_ex(val_data_example, args.modality, val=True)
            print("Exporting train example images...", flush=True)
            val_data_example = train_dataset[0] #train_ds randcropbyposneglabel return a lists
            export_ex(val_data_example, args.modality, val=False)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                drop_last=True,
                                #num_workers=4, #spawn error
                                collate_fn=pad_list_data_collate)
        print(f"train dataloader length: {len(train_loader)}", flush=True) #84 batches
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                #num_workers=4,
                                collate_fn=pad_list_data_collate)
        print(f"val dataloader length  : {len(val_loader)}", flush=True) #21 batches
        
        end_time_data = time.time() # 87s if 4 workers
        print(f"Data loading time: {end_time_data - start_time_epoch:2.5f} seconds", flush=True)

        # Reset model, optimizer, and scaler before each fold
        base_model = UNet(
            spatial_dims = 3,  # Use 3D UNet, not slicing
            in_channels = 1,  # PET, CT
            out_channels = args.out_channels,  # 0: background, 1: cancer, 2: lymphnode #HECKTOR
            channels=(64, 128, 256),  # Number of features at each layer #(16, 32, 64, 128, 256)
            strides=(2, 2)  # Downsampling steps
        ).to(device)
        model = UNet_Segmenter(base_model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=args.lr, 
                                      weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler()

        
        # Training and Validation for each fold
        print(time.ctime(), flush=True)
        best_metric = -1
        best_epoch = -1
        
        epoch_loss_values = []
        train_dices = []
        val_dices = []
        # Early stopping parameters
        counter = 0
        early_stop = False
        skip_fold = False
        final_epoch = args.max_epochs

        #what if resume training?
        bestval_save_dir = os.path.join(model_dir, F"best_model_fold_{fold}.pth") #best #last_model_fold_{fold}.pth
        last_save_dir = os.path.join(model_dir, f"last_model_fold_{fold}.pth")
        if os.path.exists(last_save_dir):
            checkpoint = torch.load(last_save_dir)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            final_epoch = checkpoint['epoch'] #final_epoch, not args here
            best_epoch = checkpoint['best_epoch']
            epoch_loss_values = checkpoint['epoch_loss_values']
            best_metric = checkpoint['best_metric_value'] #value here
            train_dices = checkpoint['train_dices']
            val_dices = checkpoint['val_dices']
            skip_fold = True
            #debug
            print(f"len(epoch_loss_values): {len(epoch_loss_values)}", flush=True)
            print(f"Loaded model from {last_save_dir}, epoch {final_epoch}", flush=True)
            print(f"Skipping fold {fold+1}/{args.k_folds}, checkpoint already exists.", flush=True)
        elif os.path.exists(bestval_save_dir):
            checkpoint = torch.load(bestval_save_dir)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            args.min_epochs = checkpoint['epoch'] + 1 #ex. best epoch is 30, finished. start training from ep31
            best_epoch = checkpoint['best_epoch']
            epoch_loss_values = checkpoint['epoch_loss_values']
            best_metric = checkpoint['best_metric_value'] #value here
            train_dices = checkpoint['train_dices']
            val_dices = checkpoint['val_dices']
            #debug
            print(f"len(epoch_loss_values): {len(epoch_loss_values)}", flush=True)
            print(f"Loaded model from {bestval_save_dir}, epoch {args.min_epochs}", flush=True)
        else:
            print("Best or last model not found. Start training.", flush=True)
            args.min_epochs = 0

        print("Training...", flush=True)
        for epoch in range(args.min_epochs, args.max_epochs):
            if skip_fold:
                break
            if early_stop:
                final_epoch = epoch #epoch range(0, final_epoch)
                #debug
                print(f"Final epoch: {final_epoch}", flush=True)
                print(f"len(epoch_loss_values): {len(epoch_loss_values)}", flush=True)
                print(f"Early stopping triggered. Stopping training for fold {fold+1}/{args.k_folds}", flush=True)
                break
            model.train()
            epoch_loss = 0
            idx = 0
            start_time_epoch = time.time()
            for batch_data in train_loader:
                optimizer.zero_grad()
                inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
                #debug: check if loading problem or model problem
                idx += 1

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


            if epoch % 10 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'epoch_loss_values': epoch_loss_values,
                    'val_dices': val_dices,
                    'train_dices': train_dices,
                    'best_metric_value': best_metric,
                }, os.path.join(model_dir, f"fold_{fold}_epoch_{epoch}.pth"))
                print("Epoch checkpoint saved", flush=True)
                
            # Validation
            model.eval()
            print("Compute Val dice...", flush=True)
            mean_val_dice = validation(model, val_loader, dice_metric, post_pred, post_label)
            print("Compute Train dice...", flush=True)
            mean_train_dice = validation(model, train_loader, dice_metric, post_pred, post_label)
            train_dices.append(mean_train_dice)
            val_dices.append(mean_val_dice)
            # end_time_epoch_val = time.time()
            # print(f"Epoch Val time: {end_time_epoch_val - end_time_epoch_training:2.5f} seconds", flush=True) 

            # Save best model for this fold
            # Early stopping logic
            if mean_val_dice > best_metric:
                best_metric = mean_val_dice
                best_epoch = epoch
                #should i save scalers? #go back to check mobilevit
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'epoch_loss_values': epoch_loss_values,
                    'val_dices': val_dices,
                    'train_dices': train_dices,
                    'best_metric_value': best_metric,
                }, bestval_save_dir)
                # print("Model saved!" , flush=True)
                counter = 0  # Reset the counter if validation dice improves
            else:
                counter += 1
                print(f"Early stopping counter: {counter} out of {args.patience}")
                if counter >= args.patience:
                    early_stop = True  # Trigger early stopping

            if epoch % 10 == 0:
                model.eval()
                export_ex_seg_train_val(model, f"{model_name}_fold{fold}", graph_dir, 
                                        train_loader, val_loader, epoch,
                                        dice_metric, post_pred, post_label,
                                        train_i_stop = 0, val_i_stop = args.val_i_stop, 
                                        i = 0, slice_num = args.roi_x//2, save_nifti=False) #save_nifti to check output classes

        
            end_time_epoch = time.time()
            print(f"Epoch Total time: {end_time_epoch - start_time_epoch:2.5f} seconds", flush=True) 
            print(f"Epoch {epoch+1}/{args.max_epochs} - Fold {fold+1}/{args.k_folds}: Train Loss {epoch_loss:.4f}, Train Dice {mean_train_dice:.4f}, Val Dice {mean_val_dice:.4f}", flush=True)


        # Save last model for this fold
        #checkpoint?
        if not skip_fold:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'epoch': final_epoch,
                'best_epoch': best_epoch,
                'epoch_loss_values': epoch_loss_values,
                'val_dices': val_dices,
                'train_dices': train_dices,
                'best_metric_value': best_metric,
            }, last_save_dir)
            print("Last model saved!", flush=True)

        print(f"Best validation dice score for fold {fold+1}/{args.k_folds}: {best_metric:.4f} at epoch {best_epoch}, with loss: {epoch_loss_values[-1]}", flush=True)



        # save csv
        print("fold: ", fold)
        epochs_range = range(0, final_epoch) #should I +1? or +2? i dont know why!


        print("final epoch", final_epoch)
        print("best  epoch", best_epoch) 
        print("len epoch range: ", len(epochs_range))
        print("len(epoch_loss_values): ", len(epoch_loss_values)) 
        print("len(train_dices)", len(train_dices))
        print("len(val_dices)", len(val_dices),flush=True)
        df = pd.DataFrame({
            'epoch': epochs_range,
            'train_loss': epoch_loss_values,
            'train_dice': train_dices,
            'val_dice': val_dices
        })
        df.to_csv(f'{model_dir}/train_val_loss_{model_name}_fold{fold}.csv')
        print(f"Saved training and validation metrics for fold {fold+1}/{args.k_folds}", flush=True)

        # Plot loss and dice scores
        plt.ioff()
        plt.figure(figsize=(12, 6))
        # Plot Train Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, epoch_loss_values, label='Train Loss')
        plt.title(f'Train Loss for Fold {fold+1}/{args.k_folds}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Dice Scores (Train and Val)
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_dices, label='Train Dice')
        plt.plot(epochs_range, val_dices, label='Val Dice')
        plt.title(f'Dice Scores for Fold {fold+1}/{args.k_folds}')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.ylim(0, 1)
        plt.legend()
        
        plt.tight_layout()
        plt.suptitle(f"Result from {fold+1}/{args.k_folds} folds")
        plt.savefig(f'{model_dir}/train_val_loss_{model_name}_fold{fold}.png')

        # Save eximage
        model.eval()
        export_ex_seg_train_val(model, f"{model_name}_fold{fold}", graph_dir, 
                                train_loader, val_loader, final_epoch,
                                dice_metric, post_pred, post_label,
                                train_i_stop = 0, val_i_stop = args.val_i_stop, 
                                i = 0, slice_num = args.roi_x//2)


        # Append results to the lists for all folds
        # all_epoch_loss_values.append(epoch_loss_values)
        # all_train_dices.append(train_dices)
        # all_val_dices.append(val_dices)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training finished! Total time: {total_time:2.5f} seconds", flush=True)  #not helpful for resume training


    # Check average

    # all_epoch_loss_values_np = np.array(all_epoch_loss_values)
    # all_train_dices_np = np.array(all_train_dices)
    # all_val_dices_np = np.array(all_val_dices)

    # # Is this the correct way?
    # # Average the metrics across folds
    # average_epoch_loss = np.mean(all_epoch_loss_values_np, axis=0)
    # average_train_dices = np.mean(all_train_dices_np, axis=0)
    # average_val_dices = np.mean(all_val_dices_np, axis=0)

    # # Optionally, you can also compute standard deviation for further analysis
    # std_epoch_loss = np.std(all_epoch_loss_values_np, axis=0)
    # std_train_dices = np.std(all_train_dices_np, axis=0)
    # std_val_dices = np.std(all_val_dices_np, axis=0)

    # # Print or log the averaged results
    # print("Average Loss Across Folds:", average_epoch_loss[-1])
    # print("Average Train Dices Across Folds:", average_train_dices[-1])
    # print("Average Validation Dices Across Folds:", average_val_dices[-1])



if __name__ == '__main__': #running the script
    main()