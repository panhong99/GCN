import math
import random
import numpy as np
import torch
import copy
import torchvision.transforms as transforms
from PIL import Image

class AverageMeter(object):
  def __init__(self):
    self.val = None
    self.sum = None
    self.cnt = None
    self.avg = None
    self.ema = None
    self.initialized = False

  def update(self, val, n=1):
    if not self.initialized:
      self.initialize(val, n)
    else:
      self.add(val, n)

  def initialize(self, val, n):
    self.val = val
    self.sum = val * n
    self.cnt = n
    self.avg = val
    self.ema = val
    self.initialized = True

  def add(self, val, n):
    self.val = val
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
    self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
  pred = np.asarray(pred, dtype=np.uint8).copy()
  mask = np.asarray(mask, dtype=np.uint8).copy()

  # 255 -> 0
  
  # 이게 아마 range가 0-255니까 +1을 하면 padding값이 0이 됨
  # 1이 배경, 2가 비행기 뭐 이런식으로 변경
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)

def preprocess(image, mask, color_mask, flip=False , crop=None, train = False, MI = False):
# def preprocess(image, mask , process_value, process, flip=False , crop=None):

  if flip:
    if random.random() < 0.5:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)
      mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
      # color_mask = color_mask.transpose(Image.FLIP_LEFT_RIGHT)

  data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  
  unnorm_image = copy.deepcopy(image)

  # PIL to numpy
  image = np.array(image)
  mask = np.array(mask)
    
  if crop:
    h, w = image.shape[0], image.shape[1]

    # modify image size
    H = max(0, crop[0] - h)
    W = max(0, crop[1] - w)

    image = np.pad(image,
                    pad_width=((0, H), (0, W), (0, 0)),
                    mode='constant',
                    constant_values=0)

    mask = np.pad(mask,
                  pad_width=((0, H), (0, W)),
                  mode='constant',
                  constant_values=0)    
      
  image = data_transforms(image)
  mask = torch.LongTensor(mask.astype(np.int64))

  if train:
    return image, mask

  elif train == False and MI == True:
    return image, mask

  else:
    return image, mask, unnorm_image, color_mask, H, W

  # return image, mask