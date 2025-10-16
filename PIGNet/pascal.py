from __future__ import print_function
import cv2
import torch.utils.data as data
import os
from PIL import Image,ImageOps
import numpy as np
import scipy.ndimage as ndi
import random
from utils import preprocess
import copy
import torch
import math

seed_value = 42 
random.seed(seed_value)
torch.manual_seed(seed_value)

class VOCSegmentation(data.Dataset):
  CLASSES = [
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
      'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
      'tv/monitor'
  ]

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None , process=None ,process_value=None,overlap_percentage=None,pattern_repeat_count=None):
    self.root = root
    _voc_root = os.path.join(self.root, 'VOC2012')
    _list_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size
    self.process = process
    self.process_value = process_value
    self.overlap_percentage = overlap_percentage
    self.pattern_repeat_count = pattern_repeat_count
    self.dataset_name = "pascal"
    if download:
      self.download()

    if self.train:
      _list_f = os.path.join(_list_dir, 'train_aug.txt')
    else:
      _list_f = os.path.join(_list_dir, 'val.txt')

    if self.process == 'overlap':
      print("!! process model")

    elif self.process == 'zoom':
      print("!! zoom_in model")

    self.images = []
    self.masks = []
    self.color_masks = []
    with open(_list_f, 'r') as lines:
      for line in lines:
        img_id = line.strip()

        img_path = os.path.join(_voc_root , "JPEGImages" , img_id + ".jpg")
        mask_path = os.path.join(_voc_root , "SegmentationClassAug" , img_id + ".png")
        color_mask_path = os.path.join(_voc_root , "SegmentationClass" , img_id + ".png")

        # _image = _voc_root + line.split()[0]
        # _mask = _voc_root + line.split()[1]

        assert os.path.isfile(img_path)
        assert os.path.isfile(mask_path)

        self.images.append(img_path)
        self.masks.append(mask_path)
        self.color_masks.append(color_mask_path)

  def __getitem__(self, index):

    _img = Image.open(self.images[index]).convert('RGB')
    _target = Image.open(self.masks[index])

    _color_target  = Image.open(self.color_masks[index]).convert('RGB')

    if self.process != None:
        _target = _target.convert("L")

    # add image process for test
    if self.process == 'zoom':
      _img, _target, _color_target = self.zoom_center(_img, _target, _color_target, self.process_value)

    elif self.process == 'overlap' and index < len(self.images) - 1:

      next_img=Image.open(self.images[index+1]).convert('RGB')
      next_target=Image.open(self.masks[index+1])
      next_color_target=Image.open(self.color_masks[index+1]).convert('RGB')

      _img, _target, _color_target = self.overlap(_img, _target,_color_target, next_img, next_target, next_color_target, self.overlap_percentage)
      
      if _img==None:
        return None,None,None,None

    elif self.process == 'repeat':
      _img, _target, _color_target = self.repeat(_img, _target, _color_target, self.pattern_repeat_count)
      if _img==None:

        return None,None,None,None
      
    else: # train
      _img, _target, _color_target = self.image_resizing(_img, _target, _color_target)

      if _img == None:
        return None, None, None, None

    _img, _target, unnorm_image, _color_target = preprocess(_img, _target, _color_target,self. dataset_name, self.process_value, self.process,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size))

    if self.transform is not None:
      _img = self.transform(_img)

    if self.target_transform is not None:
      _target = _target.unsqueeze(0)
      _target = self.target_transform(_target)
      
    return _img, _target, unnorm_image, _color_target
  
  def __len__(self):
    return len(self.images)

  def download(self):
    raise NotImplementedError('Automatic download not yet implemented.')
  
  def image_resizing(self, image, mask, color_mask, flip=True, scale=(0.5, 2.0)):

    if flip:
      if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        color_mask = color_mask.transpose(Image.FLIP_LEFT_RIGHT)

    if scale:
      w, h = image.size
      rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
      random_scale = math.pow(2, rand_log_scale)
      new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
      image = image.resize(new_size, Image.Resampling.LANCZOS)
      mask = mask.resize(new_size, Image.Resampling.NEAREST)
      color_mask = color_mask.resize(new_size, Image.Resampling.NEAREST)
    
    h, w = image.size[:2] # image type is PIL image
    scale = self.crop_size / max(h, w)
    
    new_h = int(scale * h)
    new_w = int(scale * w)
      
    new_image = image.resize((new_h, new_w), Image.Resampling.LANCZOS)
    new_mask = mask.resize((new_h, new_w), Image.Resampling.NEAREST)
    new_color_mask = color_mask.resize((new_h, new_w), Image.Resampling.NEAREST)
      
    return new_image, new_mask, new_color_mask

  def find_contours(self, mask):
    mask_array = np.array(mask)
    _, binary_mask = cv2.threshold(mask_array, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

  def extract_inner_region(self, image, contour):
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, mask)
    return result

  def overlap(self, image, mask, color_mask, next_image, next_mask, next_color_mask, overlap_percentage=0.5):
      numpy_image = np.array(image)
      original_opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

      next_numpy_image = np.array(next_image)
      next_original_opencv_image = cv2.cvtColor(next_numpy_image, cv2.COLOR_RGB2BGR)

      contour1 = self.find_contours(mask)
      if len(contour1) != 1:
          return None, None, None 
          
      inner_region1 = self.extract_inner_region(original_opencv_image, contour1[0])
      inner_image1 = Image.fromarray(cv2.cvtColor(inner_region1, cv2.COLOR_BGR2RGB))

      contour2 = self.find_contours(next_mask)
      if len(contour2) != 1:
          return None, None, None 
          
      inner_region2 = self.extract_inner_region(next_original_opencv_image, contour2[0])
      inner_image2 = Image.fromarray(cv2.cvtColor(inner_region2, cv2.COLOR_BGR2RGB))
      
      color_mask_np = np.array(color_mask)
      next_color_mask_np = np.array(next_color_mask)

      inner_image1_np = np.array(inner_image1)
      inner_image2_np = np.array(inner_image2)
      inner_mask1_np = np.array(mask)
      inner_mask2_np = np.array(next_mask)

      overlap_width = int(inner_image1_np.shape[1] * overlap_percentage)

      canvas_width = inner_image1_np.shape[1] + inner_image2_np.shape[1] - overlap_width
      canvas_height = max(inner_image1_np.shape[0], inner_image2_np.shape[0])
      canvas_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
      canvas_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
      
      canvas_color_mask = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

      canvas_image[:inner_image1_np.shape[0], :inner_image1_np.shape[1]] = inner_image1_np
      canvas_mask[:inner_mask1_np.shape[0], :inner_mask1_np.shape[1]] = inner_mask1_np
      canvas_color_mask[:color_mask_np.shape[0], :color_mask_np.shape[1]] = color_mask_np

      start_x = inner_image1_np.shape[1] - overlap_width
      for i in range(inner_image2_np.shape[0]):
          for j in range(inner_image2_np.shape[1]):
              if canvas_mask[i, start_x + j] == 0:
                  canvas_image[i, start_x + j] = inner_image2_np[i, j]
                  canvas_mask[i, start_x + j] = inner_mask2_np[i, j]
                  canvas_color_mask[i, start_x + j] = next_color_mask_np[i, j]

      result_image = Image.fromarray(canvas_image)
      result_mask = Image.fromarray(canvas_mask)
      result_color_mask = Image.fromarray(canvas_color_mask)

      result_image, result_mask, result_color_mask = self.image_resizing(result_image, result_mask, result_color_mask)

      return result_image, result_mask, result_color_mask

  def repeat(self,image, mask, color_mask, pattern_repeat_count):
    image_size = image.size
    numpy_image = np.array(image)
    original_opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    contour1 = self.find_contours(mask)
    if len(contour1) != 1:
        return None, None, None

    inner_region1 = self.extract_inner_region(original_opencv_image, contour1[0])
    inner_image1 = Image.fromarray(cv2.cvtColor(inner_region1, cv2.COLOR_BGR2RGB))
    inner_image1_resize = inner_image1.resize(
        (image_size[0], image_size[1])
    )

    numpy_mask = np.array(mask)
    inner_mask1 = self.extract_inner_region(numpy_mask, contour1[0])
    inner_mask1_resize = Image.fromarray(inner_mask1).resize(
        (image_size[0], image_size[1])
    )

    numpy_color_mask = np.array(color_mask)
    inner_color_mask1 = self.extract_inner_region(numpy_color_mask, contour1[0])
    inner_color_mask1_resize = Image.fromarray(inner_color_mask1).resize(
        (image_size[0], image_size[1])
    )

    new_image_size = (image_size[0] * int(pattern_repeat_count), image_size[1] * int(pattern_repeat_count))
    new_image = Image.new('RGB', new_image_size)
    new_mask = Image.new('L', new_image_size)
    new_color_mask = Image.new('RGB', new_image_size)

    for i in range(pattern_repeat_count):
        for j in range(pattern_repeat_count):
            new_image.paste(inner_image1_resize, (i * image_size[0], j * image_size[1]))
            new_mask.paste(inner_mask1_resize, (i * image_size[0], j * image_size[1]))
            new_color_mask.paste(inner_color_mask1_resize, (i * image_size[0], j * image_size[1]))
          
    new_image, new_mask, new_color_mask = self.image_resizing(new_image, new_mask, new_color_mask)

    return new_image, new_mask, new_color_mask
    
  def zoom_center(self, image, mask, color_mask, zoom_factor):
    """
    Zooms into or out of the image and mask around the center by the given zoom_factor.
    Keeps the image size unchanged.
    """
    width, height = image.size

    if zoom_factor > 1:
      # Zoom in
      new_width = int(width / zoom_factor)
      new_height = int(height / zoom_factor)

      # Center coordinates
      center_x, center_y = width // 2, height // 2

      # Calculate the crop box
      left = max(center_x - new_width // 2, 0)
      right = min(center_x + new_width // 2, width)
      top = max(center_y - new_height // 2, 0)
      bottom = min(center_y + new_height // 2, height)

      # Crop the image and mask, then resize back to the original dimensions
      image = image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)
      mask = mask.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.NEAREST)
      color_mask = color_mask.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.NEAREST)

    elif zoom_factor < 1:
      # Zoom out
      new_width = int(width * zoom_factor)
      new_height = int(height * zoom_factor)

      # Resize the image and mask to the new dimensions
      resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
      resized_mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
      resized_color_mask = color_mask.resize((new_width, new_height), Image.Resampling.NEAREST)

      # Create a new black image and paste the resized image and mask in the center
      new_image = Image.new('RGB', (width, height), (0, 0, 0))
      new_mask = Image.new('L', (width, height), 255)
      new_color_mask = Image.new('RGB', (width, height), (0, 0, 0))

      new_image.paste(resized_image, ((width - new_width) // 2, (height - new_height) // 2))
      new_mask.paste(resized_mask, ((width - new_width) // 2, (height - new_height) // 2))
      new_color_mask.paste(resized_color_mask, ((width - new_width) // 2, (height - new_height) // 2))

      image = new_image
      mask = new_mask
      color_mask = new_color_mask

    image, mask, color_mask = self.image_resizing(image, mask, color_mask)

    return image, mask, color_mask