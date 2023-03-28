## Defining Corner Regression Model

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
import torch.nn as nn

import random
import yolov5
import paddleocr
from paddleocr import PaddleOCR,draw_ocr
from ensemble_boxes import *
import re
import torch.nn.functional as F
import copy



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('resnet34', pretrained=True, num_classes=768)
        self.ll = nn.Linear(768,8)
    
    def forward(self, img):
        x = self.model(img)
        x = self.ll(x)

        return x

model_reg = CNN()


## Define UNET++ Model

ENCODER = 'resnext101_32x8d'
ENCODER_WEIGHTS = 'imagenet'

model_unet = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=1,
    in_channels=3,
    activation=None,
)

preprocessing_unet = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

## Loading weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_reg = model_reg.to(device)
model_unet = model_unet.to(device)
root = '/kaggle/input/'
model_reg.load_state_dict(torch.load(root + '/weights/corner_reg.pt', map_location=device))
model_unet.load_state_dict(torch.load(root + '/weights/unetplusplus_weights.pt', map_location=device))

## Helper Functions

def get_preprocessing(preprocessing_fn):
    
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)


def to_tensor(x, **kwargs):
      
    return x.transpose(2, 0, 1).astype('float32')


def get_contour_from_mask(img):

    """
    Args:
        img (np.array): mask for which bounding box has to be formed
    Returns:
        cnr (np.array): corners of the bounding box
    """

    assert img.ndim == 2
    h, w = img.shape[:2]
    img = (img>0.9*img.max()) * 255
    img = np.ascontiguousarray(img, dtype=np.uint8)

    im_floodfill = img.copy()
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = img | im_floodfill_inv
    
    im_open = cv2.morphologyEx(np.uint8(im_out), cv2.MORPH_OPEN, np.ones((5, 5)),iterations= 5)
    image_sharp = cv2.morphologyEx(im_open, cv2.MORPH_CLOSE, np.ones((5, 5)),iterations= 5)

    cnts = cv2.findContours(image_sharp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    cnt = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    cnr = np.zeros((4,2))
    cnr[0] = cnt[(cnt[:,:,0] + cnt[:,:,1]).argmin()][0]
    cnr[1] = cnt[(cnt[:,:,0] - cnt[:,:,1]).argmax()][0]
    cnr[2] = cnt[(cnt[:,:,0] + cnt[:,:,1]).argmax()][0]
    cnr[3] = cnt[(cnt[:,:,1] - cnt[:,:,0]).argmax()][0]
    
    return cnr

def corner_regression(img_path, model, size = 224):

    """
    Args:
        img_path (str): path to the image
        model (torch.nn.Module): model for corner regression
        size (int): size of the image to be fed to the model
    Returns:
        corner_preds (np.array): corners of the bounding box of mask
    """
    
    model.eval()
    img_orig = cv2.imread(img_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    img = A.Compose([A.Resize(size, size)])(image = img_orig)["image"]
    
    img = (np.transpose(img, (2, 0, 1))) / 255.0
    img = torch.tensor(img[np.newaxis,:,:,:])
    
    with torch.no_grad():
        corner_preds = model(img.to(device, dtype=torch.float32))
        
    corner_preds = corner_preds.detach().cpu().numpy()
    corner_preds = np.float32(corner_preds.reshape((4,2)))

    for pt in corner_preds:
        pt[0] = pt[0]/size * 1280.0
        pt[1] = pt[1]/size * 720.0

    width, height = 1280, 720
    target_corners = np.array([(0, 0), (width, 0), (width, height), (0, height)])
    
    H, _ = cv2.findHomography(corner_preds, target_corners, params=None)
    
    transformed_image = cv2.warpPerspective(
        img_orig, H, (img_orig.shape[1], img_orig.shape[0]))
    
    return corner_preds

def unet_prediction(img_path, model, preprocessing_fn, size = 320):

    """
    Args:
        img_path (str): path to the image
        model (torch.nn.Module): model for segmentation
        preprocessing_fn (callbale): data normalization function
        size (int): size of the image to be fed to the model
    Returns:
        corners (np.array): corners of the bounding box of mask
    """
    
    model.eval()
    im2 = np.array(Image.open(img_path))
    h, w, n = im2.shape
    true = im2.copy()
    resize_img = A.Compose([
            A.Resize(size, size)
        ])

    preprocessor = get_preprocessing(preprocessing_fn)
    
    im2 = resize_img(image = im2)['image']
    im = im2.copy()
    im2 = preprocessor(image = im2)['image']
    
    img = torch.tensor(im2)
    
    with torch.no_grad():
        out = model(img.unsqueeze(0).to(device, dtype = torch.float32))
        
    img = img.numpy()
    out = out.sigmoid().detach().cpu().numpy()
    temp = np.transpose(out.squeeze(0), [1,2,0]).copy()
    
    corners = get_contour_from_mask(np.uint8(temp*255).squeeze())
    
    for pt in corners:
        pt[0] = pt[0]/size * w*1.0
        pt[1] = pt[1]/size * h*1.0
        
    return corners

def min_max_corner_fusion(corner_unet, corner_reg):

    """
    Args:
        corner_unet (np.array): corners of the bounding box from segmentation
        corner_reg (np.array): corners of the bounding box from regression
    Returns:
        corner_min_max (np.array): corners of the bounding box from Min Max Corner Fusion Algorithm
    """
    
    corner_min_max = np.zeros((4,2))
        
    corner_min_max[0][0] = min(corner_unet[0][0], corner_reg[0][0])
    corner_min_max[0][1] = min(corner_unet[0][1], corner_reg[0][1])
    corner_min_max[1][0] = max(corner_unet[1][0], corner_reg[1][0])
    corner_min_max[1][1] = min(corner_unet[1][1], corner_reg[1][1])
    corner_min_max[2][0] = max(corner_unet[2][0], corner_reg[2][0])
    corner_min_max[2][1] = max(corner_unet[2][1], corner_reg[2][1])
    corner_min_max[3][0] = min(corner_unet[3][0], corner_reg[3][0])
    corner_min_max[3][1] = max(corner_unet[3][1], corner_reg[3][1])
    
    return corner_min_max

def screen_extraction(img_path, model_unet, model_reg, preprocessing_unet, mode = 'accurate'):

    """
    Args:
        img_path (str): path to the image
        model_unet (torch.nn.Module): unet model
        model_reg (torch.nn.Module): corner regression model
        preprocessing_unet (callbale): data normalization function of unet
        mode (str): mode of the algorithm
    Returns:
        transformed_image (np.array): transformed image
    """
    
    img = cv2.imread(img_path)
    
    if mode == 'accurate':
        
        corner_unet = unet_prediction(img_path, model_unet, preprocessing_unet)
        corner_reg = corner_regression(img_path, model_reg)
        corner_preds = min_max_corner_fusion(corner_unet, corner_reg)
        
    else:
        corner_preds = corner_regression(img_path, model_reg)
    
    for pt in corner_preds:
        pt[0] = pt[0]/img.shape[1] * 1280.0
        pt[1] = pt[1]/img.shape[0] * 720.0
        
        
    width, height = 1280, 720
    target_corners = np.array([(0, 0), (width, 0), (width, height), (0, height)])

    # Get matrix H that maps source_corners to target_corners
    H, _ = cv2.findHomography(corner_preds, target_corners, params=None)

    # Apply matrix H to source image.
    transformed_image = cv2.warpPerspective(
        img, H, (img.shape[1], img.shape[0]))
    
#     plt.imshow(transformed_image)
    
    return transformed_image