## Loading models for detection
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = '/kaggle/input/'
yolo_model_det = yolov5.load(root + '/weights/final_yolo_weights.pt')
paddle_ocr_det_acc = PaddleOCR(cpu_threads=1, rec_batch_num=2, rec_algorithm='CRNN', rec_image_inverse=False)
model_ocr = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)
yolo_fast = yolov5.load(root + 'weights/yolo_on_6_fast.pt')
paddle_fast = PaddleOCR(use_angle_cls=False, lang='en', ocr_version = 'PP-OCR', structure_version = 'PP-Structure', 
                rec_algorithm = 'CRNN', max_text_length = 200, use_space_char = False, lan = 'en', det = False,
                cpu_threads = 12, cls = False,use_gpu=False )

yolo_model_det.eval();
model_ocr.eval();
yolo_fast.eval();

## Helper Functions
def return_fast_output(yolo_model, img):

    image = img.copy()

    results_yolo = yolo_model(img)

    try:
        boxes = results_yolo.pred[0][:, :4].tolist()
        scores = results_yolo.pred[0][:, 4].tolist()
        labels = results_yolo.pred[0][:, 5].tolist()
    except:
        boxes = []
        scores_yolo = []
        labels_yolo = []
    
    dic = {}
    for each in labels:
        if each not in dic.keys():
            dic[each] = (0,[])
    
    for i in range(len(labels)):
        score , box = dic[labels[i]]
        if score < scores[i]:
            dic[labels[i]] = (scores[i], boxes[i])
    
    # print(dic)
    return dic

def recognize_fast(image,dic,rec):

    vitals = {}
    labels = {0.0: 'DBP' , 1.0:'HR' , 2.0:'MAP', 3.0:'RR' ,4.0:'SBP' ,5.0:'SPO2' }
    for each in dic.keys():
        score, box = dic[each]
        xmin = int(box[0])
        xmax = int(box[2])
        ymin = int(box[1])
        ymax = int(box[3])
        img = image[ymin:ymax,xmin:xmax]
        text = rec.ocr(img,cls = False,det = False)[0][0][0]
        text = text.replace('(','').replace(')','').replace('/','').replace('-','').replace('*','')
        if text.isdigit():
            vitals[labels[each]] = text

    return vitals

def return_output(yolo_model, paddle_ocr, img):
  image = img.copy()
  results_yolo = yolo_model(img)
  try:
    boxes = results_yolo.pred[0][:, :4].tolist()
    scores_yolo = results_yolo.pred[0][:, 4].tolist()
    labels_yolo = results_yolo.pred[0][:, 5].tolist()
  except:
    boxes = []
    scores_yolo = []
    labels_yolo = []
  boxes_yolo = []
  for box in boxes:
    boxes_yolo.append([box[0]/1280, box[1]/720, box[2]/1280, box[3]/720])
  # results_paddle = paddle_ocr.ocr(img, cls=False, rec = True)
  # final_boxes_paddle = []
  # final_confidence_paddle = []
  # final_labels_paddle = [] 
  # for i in range(len(results_paddle[0])):
  #   xmin = results_paddle[0][i][0][0][0]/1280.
  #   ymin = results_paddle[0][i][0][0][1]/720.
  #   xmax = results_paddle[0][i][0][2][0]/1280.
  #   ymax = results_paddle[0][i][0][2][1]/720.
  #   temp = [xmin, ymin, xmax, ymax]

  #   # area
  #   if (xmax*1280 - xmin*1280)*(ymax*720 - ymin*720) < 1000:
  #     continue
    
  #   if (xmax*1280 - xmin*1280) < 120:
  #     continue
  #   if (xmax*1280 - xmin*1280) > 300:
  #     continue
  #   final_boxes_paddle.append(temp)
  #   final_confidence_paddle.append(results_paddle[0][i][1][1])
  # for i in range(len(final_confidence_paddle)):
  #   final_labels_paddle.append(0)
  result_box = boxes_yolo
  result_conf = scores_yolo
  result_label = labels_yolo
  return result_box, result_conf, result_label, img
  
################################################################################

def wbf_ensemble(boxes_list, scores_list, labels_list, image):
  weights = [2, 1]
  iou_thr = 0.6
  skip_box_thr = 0.01
  sigma = 0.1
  boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
  return boxes, scores

def recognize(image,boxes,scores):
    imgs = []
    for box in boxes:
        xmin = int(box[0]*1280)
        ymin = int(box[1]*720)
        xmax = int(box[2]*1280)
        ymax = int(box[3]*720)
        img = image[ymin:ymax,xmin:xmax]
        imgs.append(img)
    
    procs = [preproc_image(img) for img in imgs]
    preds = model_ocr(torch.cat(procs, dim=0))
    labels = inference_pred(preds)
    
    return labels,image,boxes,scores

def preproc_image(img):
    img = Image.fromarray(img).convert('RGB')
    transform = T.Compose([
            T.Resize((32, 128)),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])
    img = transform(img)
    return img.unsqueeze(0)

def inference_pred(pred):
    pred = pred.softmax(-1)
    label, _ = model_ocr.tokenizer.decode(pred)
    return label

def check_string(string):
    if(('(' in string or ')' in string) and '/' in string):
        return False
    elif (string.count('(') > 1 or string.count('/') > 1 or string.count(')') > 1):
        return False
    string = string.replace('(', '').replace('/', '').replace(')', '')
    pattern = r'^[\d]+$'
    return re.match(pattern, string) != None

#################


def image_dict(text , boxes , scores, image):
  c = 0
  text_l = []
  boxes_l = []
  scores_l = []

  for i, t in enumerate(text):
    if check_string(t):
      text_l.append(t)
      boxes_l.append(boxes[i])
      scores_l.append(scores[i])

  boxes_l, scores_l = np.array(boxes_l), np.array(scores_l)
  nums = np.array([float(txt.replace('(', '').replace('/', '').replace(')', '')) for txt in text_l])
  try:
    ind = np.argsort(scores_l)[-6:]
    scores_l = scores_l[ind]
    text_l = [text_l[x] for x in ind]
    boxes_l = boxes_l[ind]
    nums = nums[ind]
  except:
    pass
  boxes_dic = []
  for i,num in enumerate(text_l):
    bbxi = boxes_l[i]
    nm = nums[i]
    text_data = np.array([0.0, 0.0, 0.0])
    if '/' in num:
      text_data[0] = 1.0
    if '(' in num:
      text_data[1] = 1.0
    if ')' in num:
      text_data[2] = 1.0
    boxes_dic.append({'bbox': bbxi, 'num': nm, 'text_data': text_data})
    
  return {'image': image, 'val_vec': nums.tolist(), 'boxes': boxes_dic}


def number_detection(img, mode = 'accurate'):
    
  if mode == 'accurate':
    
    boxes, scores, result_label, img = return_output(yolo_model_det, paddle_ocr_det_acc, img)
    # result_box, result_conf, result_label, img = return_output(yolo_model_det, paddle_ocr_det_acc, img)

    # boxes, scores, img = wbf_ensemble(result_box, result_conf, result_label, img)

    text , img, boxes , scores = recognize(img, boxes,scores)

    number_dict = image_dict(text , boxes , scores, img)

  else:

    temp = return_fast_output(yolo_fast, img)

    number_dict = recognize_fast(img, temp, paddle_fast)

  return number_dict

