# just a backup file
import os
from os import listdir
import numpy as np
import nibabel as nib #for reading .nii.gz format MRI files
import matplotlib.pyplot as plt
import pandas as pd

import timm
import torch

model = timm.create_model("hf_hub:timm/mobilevit_xxs.cvnets_in1k", pretrained=True)

x = torch.randn(1, 3, 224, 224) #pretrained weights need 3 channels

print(model(x).shape) #torch.Size([1,1000])
