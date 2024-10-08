
print("Start running", flush=True) 

import os
from os import listdir, path
import csv
import sys
import json
# import time
import numpy as np
#import nibabel as nib #for reading .nii.gz format MRI files
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
    EnsureChannelFirstd,
    AsDiscreted, AsDiscrete, 
    Compose,
    LoadImaged,  LoadImage,
    Resized,
    Orientationd,
    SpatialCropd,
    CenterSpatialCropd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged, 
    Spacingd,
    SpatialPadd,
    EnsureTyped,
    ToTensord,
    SaveImaged
)


import argparse #args
parser = argparse.ArgumentParser(description='segmentation pipeline')
# parser.add_argument('--distributed', action='store_true', help='start distributed training') #if the --distributed flag is passed when running the script, the argument will be set to True. If the flag is not provided, the argument will default to False.
# parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
# parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
# parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--a_min_ct', default=-750, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max_ct', default=1000, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--a_min_pet', default=0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max_pet', default=10000, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=1 + 2, type=int, help='number of output channels') #0: background, 1: cancer, 2: lymph node #HECKTOR
parser.add_argument('--batch_size', default=4, type=int, help='number of batch size')
parser.add_argument('--max_epochs', default=10, type=int, help='max number of training epochs') #1 #50
# parser.add_argument('--modality', default='CT', type=str, help='modality (PET or CT)')
# parser.add_argument('--k_folds', default=5, type=int, help='folds for cross validation')
parser.add_argument('--test_size', default=0.0, type=float, help='train/test split ratio') #0.2
#test_size = 0.3
parser.add_argument('--cache_rate', default=0.1, type=float, help='CacheDataset cache_rate') #0.2
parser.add_argument('--cache_num', default=4, type=float, help='CacheDataset cache_num') #10


""" Functions:"""


""" Directory & Constants: """

#!nvcc --version
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}", flush=True)

# # distributed training:
def main():
    args = parser.parse_args() 
    main_worker(gpu=0, args=args) #not distributed

def main_worker(gpu, args):

    # Defining model name
    cwd = os.getcwd() #/data/deasy/shiqin_data/intern24/
    save_dir = os.path.join(cwd, f"preprocessed") 
    pet_dir = os.path.join(save_dir, "PET")
    ct_dir = os.path.join(save_dir, "CT")
    label_dir = os.path.join(save_dir, "label")
    if path.exists(save_dir):
        print(f"Found folder with same name: {save_dir}")
        # response = input("Do you want to use the same name? (yes/no): ").strip().lower()
        # if response == "no":
        #     print("Please change the save_dir and rerun") # later add function to change name
        #     sys.exit()
    else:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(pet_dir, exist_ok=True)
        os.makedirs(ct_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        print(f"Create new folder {save_dir}")
        print(f"Create new folder {pet_dir}")
        print(f"Create new folder {ct_dir}")
        print(f"Create new folder {label_dir}")
    print(f"model_dir: {save_dir}", flush=True)

    # Load data using the MONAI's decathlon format reader, specifically for Hecktor dataset (ct, pet, label)
    file_dir = './'
    json_file =  f"{file_dir}/train_json.json" #ex_data_CT.json
    datalist = load_decathlon_datalist(json_file, is_segmentation=True, 
                                    data_list_key="all_data", base_dir="./")
    print("length of datalist: ", len(datalist), flush=True)

    # train_datalist, test_files = train_test_split(
    #     datalist, test_size=args.test_size, random_state=42
    # ) #335,84,84 #args.test_size #debug the loss curve plotting

    # Define the transforms for training and validation sets
    def get_transforms(args):
        # if args.modality == "CT":
        #     args.a_min = args.a_min_ct
        #     args.a_max = args.a_max_ct
        # elif args.modality == "PET":
        #     args.a_min = args.a_min_pet
        #     args.a_max = args.a_max_pet
        # else:
        #     raise ValueError("modality must be either CT or PET")
        
        train_transforms = Compose([
            LoadImaged(keys=["PET", "CT", "Mask"]),
            EnsureChannelFirstd(keys=["PET", "CT", "Mask"]),
            Orientationd(keys=["PET", "CT", "Mask"], axcodes="RAS"),
            #scaling here may cause artifacts, esp. PET: [0,0.09]
            ScaleIntensityRanged(keys=["PET"], 
                                 a_min=args.a_min_pet, a_max=args.a_max_pet, 
                                 b_min=args.b_min, b_max=args.b_max, clip=True),
            ScaleIntensityRanged(keys=["CT"], 
                                 a_min=args.a_min_ct, a_max=args.a_max_ct, 
                                 b_min=args.b_min, b_max=args.b_max, clip=True),
            Spacingd(keys=["PET", "CT", "Mask"], 
                     pixdim=(args.space_x, args.space_y, args.space_z), 
                     mode=("bilinear", "bilinear", "nearest"), ),
            CropForegroundd(keys=["PET", "CT", "Mask"], 
                            source_key="Mask", 
                            allow_smaller=True, #will pad automatically?
                            margin=(64,64,64)), #hardcoded
            SpatialPadd(keys=["PET", "CT", "Mask"], 
                        spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            CenterSpatialCropd(keys=["PET", "CT", "Mask"], 
                               roi_size=(args.roi_x, args.roi_y, args.roi_z)),
            SaveImaged(
                        keys=["PET", "CT", "Mask"],
                        output_dir=save_dir,
                        output_postfix="trans",
                        resample=False,  # already spacingd above
                        separate_folder=False,  # Save files in a different folder
                    )
        ])

        return train_transforms



    # Split the dataset
    train_files = datalist #train_datalist
    print("length of train_files: ", len(train_files), flush=True)

    train_transforms = get_transforms(args)

    print("Data loading...", flush=True)
    train_dataset = CacheDataset(data=train_files, transform=train_transforms, 
                                num_workers=4,
                                cache_rate=args.cache_rate,
                                cache_num=args.cache_num)#, progress=False) #cache_num will take min
    print(f"train dataset len: {len(train_dataset)}", flush=True) #335 #debug: 

 
    for i in range(len(train_dataset)):
        data_item = train_dataset[i]  # This will apply all the transforms and save images
        if i % 10 == 0:
            print(f"Saved preprocessed image and label for item {i}", flush=True)

    print("Preprocessing and saving completed.", flush=True)
                


    # List all files in the directory
    all_files = [f for f in os.listdir(save_dir) if f.endswith(".nii.gz")]
    num_files = len(all_files)
    print(f"Number of files in {save_dir}: {num_files}", flush=True)

    # Initialize MONAI's LoadImage to load the images and metadata
    load_image = LoadImage(image_only=False)

    # Dictionary to store the results
    results = {}

    # Process each file
    for file_name in all_files:
        # Extract the base ID (first 8 characters)
        base_id = file_name[:8]
        
        # Load the image and metadata
        file_path = os.path.join(save_dir, file_name)
        image, metadata = load_image(file_path)
        
        # Extract shape and pixdim
        image_shape = image.shape
        pixdim = metadata["pixdim"][1:4]
        
        # Store the info based on the suffix (__CT, __PT, or label)
        if base_id not in results:
            results[base_id] = {"ID": base_id}
        
        if "__CT" in file_name:
            results[base_id]["CT_path"] = file_name
            results[base_id]["CT_shape"] = image_shape
            results[base_id]["CT_pixdim"] = pixdim
        elif "__PT" in file_name:
            results[base_id]["PET_path"] = file_name
            results[base_id]["PET_shape"] = image_shape
            results[base_id]["PET_pixdim"] = pixdim
        else:
            results[base_id]["label_path"] = file_name
            results[base_id]["label_shape"] = image_shape
            results[base_id]["label_pixdim"] = pixdim

    # Save the results to a CSV file
    output_csv = "preprocessed_info.csv"
    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ["ID", "CT_path", "PET_path", "label_path", "CT_shape", "PET_shape", "label_shape", "CT_pixdim", "PET_pixdim","label_pixdim" ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write the rows
        for result in results.values():
            writer.writerow(result)

    print(f"CSV file saved: {output_csv}")

    # Save a json version
    df = pd.read_csv(output_csv)
    json_output_CT = {"all_data": []}
    json_output_PET = {"all_data": []}

    for index, row in df.iterrows():
        json_output_CT["all_data"].append({
            "image": f"preprocessed/{row['CT_path']}",
            "PET": f"preprocessed/{row['PET_path']}",
            "label": f"preprocessed/{row['label_path']}"
        })
        json_output_PET["all_data"].append({
            "CT": f"preprocessed/{row['CT_path']}",
            "image": f"preprocessed/{row['PET_path']}",
            "label": f"preprocessed/{row['label_path']}"
        })
    
    output_file_path = 'train_json_CT_preprocessed.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(json_output_CT, json_file, indent=4)

    output_file_path = 'train_json_PET_preprocessed.json'
    with open(output_file_path, 'w') as json_file:
        json.dump(json_output_PET, json_file, indent=4)
    
    print(f"JSON file saved: {output_file_path}")

if __name__ == '__main__': #running the script
    main()