# just a backup file
import os
from os import listdir
import numpy as np
import nibabel as nib #for reading .nii.gz format MRI files
import matplotlib.pyplot as plt
import pandas as pd

import timm
import torch

import torchvision.transforms as trans


# Feature map
m2 = timm.create_model(
    'mobilevitv2_050.cvnets_in1k', 
    features_only=True, 
    pretrained=True,
    num_classes=0, #remove classifier 
    in_chans = 3)

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(m2)
transforms = timm.data.create_transform(**data_config, is_training=True)

# torchvision transforms module takes PIL image with 3 channels, need to add lines
#pil_image = trans.ToPILImage()(slice_T2)
#x = transforms(pil_image).unsqueeze(0) #torch.Size([1, 3, 256, 256])
#print("x.shape = {}".format(x.shape))
#output = m2(x) #list with len=5 tensors

