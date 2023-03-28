import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn, optim
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

import timm
import segmentation_models_pytorch as smp
import imutils
from skimage.transform import ProjectiveTransform

import os
from tqdm import tqdm
from PIL import Image
import albumentations as A
from sklearn.model_selection import train_test_split
import gc
import glob

import random
import yolov5
import paddleocr
from paddleocr import PaddleOCR,draw_ocr
from ensemble_boxes import *
import re
import torch.nn.functional as F
import copy

class CRABBNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=0, in_chans=4)
        self.ll = nn.Linear(2048+7,6)
        self.fl = nn.Linear(14,6)
    
    def forward(self, img, text_data, num, val_vec):
        x = self.model(img)
        x = torch.cat([x, num.view(-1, 1), self.fl(torch.cat([num.view(-1, 1), text_data.view(-1, 3), val_vec.view(-1, 10)], dim=1))], dim=1)
        x = self.ll(x)
        x = x.view(-1, 6)
#         x = F.softmax(x, dim = 1)
        
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = '/kaggle/input/'   
model_crabb = CRABBNET()
model_crabb = model_crabb.to(device)
model_crabb.load_state_dict(torch.load(root + '/weights/crabbnet.pt', map_location = device))
model_crabb.eval();

class InferDataset(Dataset):
    def __init__(self, img_dict):
        super().__init__()
        
        self.img_dict = img_dict
                          
    def __getitem__(self, idx):
        img = self.img_dict['image']
        val_vec = np.array(self.img_dict['val_vec'])
        val_vec.resize(10,)
        np.random.shuffle(val_vec)
        
        boxes = self.img_dict['boxes'][idx]  
        bbox = boxes['bbox']
        
        xmin = int(bbox[0]*1280)
        ymin = int(bbox[1]*720)
        xmax = int(bbox[2]*1280)
        ymax = int(bbox[3]*720)

        mask = np.zeros((img.shape[0], img.shape[1], 1))
        mask = cv2.rectangle((mask), (xmin, ymin), (xmax, ymax), 255, -1)

        # plt.imshow(mask[:,:,0])
        # plt.show()
        
        num = boxes['num']
        text_data = boxes['text_data']

        img = cv2.resize(img, (224, 224))
        mask = cv2.resize(mask, (224,224))[:,:,None]
                
        arr4 = np.concatenate([img, mask], axis = 2)
                   
        arr4 = (np.transpose(arr4, (2, 0, 1))) / 255.0
        arr4 = torch.tensor(arr4)
        val_vec = torch.tensor(val_vec)
                
        return arr4, torch.tensor(num), torch.tensor(text_data), val_vec
    
    def __len__(self):
        return len(self.img_dict['boxes'])

########################

def pred_organizing(pred_mat, nums):
    num_boxes = nums.shape[0]
    box_store = set(range(num_boxes))
    mat_variance = torch.var(pred_mat, dim=0)
    argmax_dim0 = torch.argmax(pred_mat, dim=0)
    sorted_, indices = torch.sort(mat_variance, descending=True)
    
    res_dict = {'HR':None, 'RR':None, 'SPO2':None, 'SBP':None, 'DBP':None, 'MAP':None}
    label_cols = ['HR', 'RR', 'SPO2', 'SBP', 'DBP', 'MAP']
    
    for ind in indices:
        
        argmax_dim0 = torch.argmax(pred_mat, dim=0)
        box_ind = argmax_dim0[ind].item()

        if box_ind not in box_store:
            present = list(box_store)
            if len(present) == 0:
                continue
            
        res_dict[label_cols[ind]] = nums[box_ind]
        pred_mat[box_ind, :] = -100000
    
        if torch.unique(pred_mat).shape[0] == 1:
            break
    
    return res_dict

############################

def final_inference(img_dict):
    test_img = InferDataset(img_dict)

    test_loader = DataLoader(
            dataset=test_img,
            batch_size = len(img_dict['boxes']),
            num_workers = 0,
    )


    imgs, nums, text_data, val_vec = next(iter(test_loader))

    imgs = imgs.float()
    nums = nums.float()
    text_data = text_data.float()
    val_vec = val_vec.float()

    with torch.no_grad():
        label_preds = model_crabb(imgs, text_data, nums, val_vec)

    nums = nums.numpy()

    yerr_dict = pred_organizing(label_preds, nums)
    yerr_dict = {k:v for k,v in yerr_dict.items() if v is not None}

    return yerr_dict

#############################