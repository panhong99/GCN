import math
import random
import numpy as np
import torch
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
  pred += 1
  mask += 1
  pred = pred * (mask > 0)

  inter = pred * (pred == mask)
  (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
  (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
  (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
  area_union = area_pred + area_mask - area_inter

  return (area_inter, area_union)

def preprocess(image, mask, dataset_name, process, process_value , flip=False, scale=None, crop=None):

  # seed = 42

  # random.seed(seed)
  # np.random.seed(seed)
  # torch.manual_seed(seed)
  
  if flip:
    if random.random() < 0.5:
      image = image.transpose(Image.FLIP_LEFT_RIGHT)
      mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
  if scale:
    w, h = image.size
    rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
    random_scale = math.pow(2, rand_log_scale)
    new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    mask = mask.resize(new_size, Image.Resampling.NEAREST)

  data_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

  # PIL to numpy
  image = np.array(image)
  mask = np.array(mask)
    
  if crop:
    h, w = image.shape[0], image.shape[1]

    # modify image size
    pad_tb = max(0, crop[0] - h)
    pad_lr = max(0, crop[1] - w)

    image = np.pad(image,
                    pad_width=((0, pad_tb), (0, pad_lr), (0, 0)),
                    mode='constant',
                    constant_values=0)

    mask = np.pad(mask,
                  pad_width=((0, pad_tb), (0, pad_lr)),
                  mode='constant',
                  constant_values=255)    
    
    h, w = image.shape[0], image.shape[1]
    
    # image crop
    if (process != "zoom"):
      i = random.randint(0, h - crop[0])
      j = random.randint(0, w - crop[1])
    
    elif (process == "zoom") and (process_value > 0.5):
      i = random.randint(0, h - crop[0])
      j = random.randint(0, w - crop[1])
    
    elif (process == "zoom") and (process_value <= 0.5):
      i = (h - crop[0]) // 2
      j = (w - crop[1]) // 2

    else:# process == None
      i = random.randint(0, h - crop[0])
      j = random.randint(0, w - crop[1])

    image = image[i:i + crop[0], j:j + crop[1],:]
    mask = mask[i:i + crop[0], j:j + crop[1]]

  # numpy to tensor
  image = data_transforms(image)
  mask = torch.LongTensor(mask.astype(np.int64))

  # if dataset_name == "pascal":
  #   vis_transforms = transforms.ToTensor()
  #   color_mask = vis_transforms(color_mask)

  # else:
  #   color_mask = torch.LongTensor(np.array(color_mask).astype(np.int64))
    
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
  
  # unnorm_image = image * std + mean

  return image, mask