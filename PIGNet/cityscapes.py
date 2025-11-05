from __future__ import print_function

import torch.utils.data as data
import os
import random
import glob
from PIL import Image
from utils import preprocess
import cv2
import numpy as np
from utils import preprocess
import copy
import torch
import math

_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
    'color': 'gtFine',
    'instance': 'gtFine'
}

_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': 'labelTrainIds',
    'color': '_gtFine_color',
    'instance': '_gtFine_instanceIds'
}

_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
    'color': 'png',
    'instance': 'png'
}

class Cityscapes(data.Dataset):
  CLASSES = [
      'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
      'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
      'truck', 'bus', 'train', 'motorcycle', 'bicycle'
  ]

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None ,
               process = None , process_value = None,  overlap_percentage = None , pattern_repeat_count = None):

    self.root = root
    self._set_local_seed(42)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size
    self.process = process
    self.process_value = process_value
    self.overlap_percentage = overlap_percentage
    self.pattern_repeat_count = pattern_repeat_count
    self.dataset_name = "cityscape"

    if download:
      self.download()

    if self.process == "overlap":
        print("!! overlap model")

    elif self.process == "zoom":
        print("!! zoom_in model")

    dataset_split = 'train' if self.train else 'val'

    self.images = self._get_files('image', dataset_split)
    self.masks = self._get_files('label', dataset_split)
    self.color_masks = self._get_files('color', dataset_split)
    self.instance_masks = self._get_files('instance', dataset_split)

  def _set_local_seed(self, seed_value):
        import random
        import numpy as np
        
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False    

  def __getitem__(self, index):

    _img = Image.open(self.images[index]).convert('RGB')
    _target = Image.open(self.masks[index])

    if self.process != None:
        _target = _target.convert("L")

    # add image process for test
    if self.process == 'zoom':
      _color_target  = Image.open(self.color_masks[index]).convert('RGB')
      _img, _target, _color_target = self.zoom_center(_img, _target, _color_target, self.process_value)

    elif self.process == 'overlap':
      _color_target  = Image.open(self.color_masks[index]).convert('RGB')
      _instance_target = Image.open(self.instance_masks[index])
      _img, _target, _color_target = self.overlap(_img, _target, _color_target, _instance_target ,self.overlap_percentage)

      if _img==None:
        return None,None,None,None,None,None

    elif self.process == 'repeat':
      _color_target  = Image.open(self.color_masks[index]).convert('RGB')
      _instance_target = Image.open(self.instance_masks[index])
      
      _img, _target, _color_target = self.repeat(_img, _target, _color_target, _instance_target, self.pattern_repeat_count)

      if _img==None:

        return None,None,None,None,None,None
      
    else: # train
      _img, _target = self.image_resizing(_img, _target)

      if _img == None:
        # return None,None,None,None,None,None
        return None,None

    _img, _target, unnorm_image, _color_target, H, W = preprocess(_img, _target, _color_target,
                               flip=True if self.train else False,
                               crop=(self.crop_size, self.crop_size))

    # _img, _target = preprocess(_img, _target, self.process_value, self.process,
    #                            flip=True if self.train else False,
    #                            crop=(self.crop_size, self.crop_size))

    if self.transform is not None:
      _img = self.transform(_img)

    if self.target_transform is not None:
      _target = _target.unsqueeze(0)
      _target = self.target_transform(_target)

      _color_target = _color_target.unsqueeze(0)
      _color_target = self.target_transform(_target)
      
    return _img, _target, unnorm_image, _color_target, H, W
    # return _img, _target

  def _get_files(self, data, dataset_split):
    pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
    search_files = os.path.join(
        self.root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)

  def __len__(self):
    return len(self.images)

  def download(self):
    raise NotImplementedError('Automatic download not yet implemented.')
  
  # def image_resizing(self, img, mask, color_mask, aug_scale=(0.5, 2.0)):
  def image_resizing(self, img, mask ,color_mask):
  
    aug_scale=(0.5, 2.0)
  
    w, h = img.size # image type is PIL image

    scale = self.crop_size / max(h, w)

    new_h = int(scale * h)
    new_w = int(scale * w)

    rand_log_scale = math.log(aug_scale[0], 2) + random.random() * (math.log(aug_scale[1], 2) - math.log(aug_scale[0], 2))
    random_scale = math.pow(2, rand_log_scale)
    new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
    image = img.resize(new_size, Image.Resampling.LANCZOS)
    mask = mask.resize(new_size, Image.Resampling.NEAREST)
    color_mask = color_mask.resize(new_size, Image.Resampling.NEAREST)

    new_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    new_mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    new_color_mask = color_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    
    return new_image, new_mask, new_color_mask
    # return new_image, new_mask

  def find_contours(self, mask, instance_mask, task=None, min_area=20000):
      instance_mask_array = np.array(instance_mask)
      semantic_mask_array = np.array(mask)

      if instance_mask_array.dtype != np.uint8:
          instance_mask_array = instance_mask_array.astype(np.int32)

      instance_ids = np.unique(instance_mask_array)
      instance_ids = instance_ids[instance_ids != 0]

      selected_contours = []

      target_classes = [2,4,5,6,7,8,9,11,12,13,14,15,17,18]

      for inst_id in instance_ids:
          mask_idx = (instance_mask_array == inst_id)

          if np.any(mask_idx):
              class_id = np.bincount(semantic_mask_array[mask_idx]).argmax()
          else:
              continue

          if class_id == 255 or class_id not in target_classes:
              continue

          inst_mask = mask_idx.astype(np.uint8) * 255
          contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

          for c in contours:
              area = cv2.contourArea(c)
              if area >= min_area:
                  selected_contours.append((c, class_id))

      if not selected_contours:
          return []

      if task == "overlap":
          class_groups = {}
          for contour, cls in selected_contours:
              class_groups.setdefault(cls, []).append(contour)

          classes = list(class_groups.keys())
          if len(classes) < 2:
              return []

          cls1, cls2 = random.sample(classes, 2)
          contour1 = random.choice(class_groups[cls1])
          contour2 = random.choice(class_groups[cls2])
          return [(contour1, cls1), (contour2, cls2)]

      else:
          return [random.choice(selected_contours)]


  def extract_inner_region(self, image, contour):
      # contour의 bounding box 구하기
      x, y, w, h = cv2.boundingRect(contour)
      roi = image[y:y+h, x:x+w]

      # 객체 mask 생성 (ROI 영역만큼)
      mask = np.zeros((h, w), dtype=np.uint8)
      cv2.drawContours(mask, [contour - [x, y]], -1, 255, thickness=cv2.FILLED)

      # RGBA 변환
      result = cv2.cvtColor(roi, cv2.COLOR_BGR2RGBA)
      result[:, :, 3] = mask

      return result

  def overlap(self, image, mask, color_mask, instance_mask, overlap_percentage=0.5):
      numpy_image = np.array(image)
      original_opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

      contour = self.find_contours(mask, instance_mask, "overlap")
      if (len(contour) == 0):
          return None, None, None

      # [(contour1, cls1), (contour2, cls2)] 
            
      cls_id1 = contour[0][1]
      cls_id2 = contour[1][1]

      contour1 = contour[0][0]
      contour2 = contour[1][0]
        
      inner_region1 = self.extract_inner_region(original_opencv_image, contour1)
      inner_region1 = Image.fromarray(inner_region1).convert("RGBA")
      
      inner_region2 = self.extract_inner_region(original_opencv_image, contour2)
      inner_region2 = Image.fromarray(inner_region2).convert("RGBA")
      
      inner_mask_np = np.array(mask)      

      inner_mask1 = self.extract_inner_region(inner_mask_np, contour1)
      inner_mask2 = self.extract_inner_region(inner_mask_np, contour2)

      inner_mask1 = Image.fromarray(inner_mask1).convert("RGBA")
      inner_mask2 = Image.fromarray(inner_mask2).convert("RGBA")

      color_mask_np = np.array(color_mask)

      inner_color_mask1 = self.extract_inner_region(color_mask_np, contour1)
      inner_color_mask2 = self.extract_inner_region(color_mask_np, contour2)

      inner_color_mask1 = Image.fromarray(inner_color_mask1[:, :, [2,1,0,3]])
      inner_color_mask2 = Image.fromarray(inner_color_mask2[:, :, [2,1,0,3]])

      w1, h1 = inner_region1.size
      w2, h2 = inner_region2.size

      overlap_ratio = overlap_percentage

      overlap_px = int(min(w1, w2) * overlap_ratio)

      canvas_w = w1 + w2 - overlap_px
      canvas_h = max(h1, h2)

      canvas_image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
      canvas_mask = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
      canvas_color_mask = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

      canvas_image.paste(inner_region1, (0, 0), inner_region1)
      canvas_mask.paste(inner_mask1, (0, 0), inner_mask1)
      canvas_color_mask.paste(inner_color_mask1, (0, 0), inner_color_mask1)

      x2 = w1 - overlap_px
      y2 = abs(h1 - h2) // 2 

      canvas_image.paste(inner_region2, (x2, y2), inner_region2)
      canvas_mask.paste(inner_mask2, (x2, y2), inner_mask2)
      canvas_color_mask.paste(inner_color_mask2, (x2, y2), inner_color_mask2)
      
      result_image = copy.deepcopy(image) 
      result_mask = copy.deepcopy(mask)
      result_color_mask = copy.deepcopy(color_mask)
      
      W, H = image.size
      w, h = canvas_image.size

      x1, y1, w1, h1 = cv2.boundingRect(contour1)
      x2, y2, w2, h2 = cv2.boundingRect(contour2)

      cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
      cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
      existing_centers = [(cx1, cy1), (cx2, cy2)]

      min_dist = int(W * 0.25)

      for _ in range(100):
          x_offset = random.randint(-W // 4, W // 4)
          y_offset = random.randint(-H // 4, H // 4)
          x = (W - w) // 2 + x_offset
          y = (H - h) // 2 + y_offset

          new_cx = x + w // 2
          new_cy = y + h // 2

          if all(((new_cx - ex)**2 + (new_cy - ey)**2)**0.5 > min_dist for ex, ey in existing_centers):
              break

      result_image.paste(canvas_image.convert("RGB"), (x, y), canvas_image.split()[3])
      result_mask.paste(canvas_mask.convert("RGB"), (x, y), canvas_mask.split()[3])
      result_color_mask.paste(canvas_color_mask.convert("RGB"), (x, y), canvas_color_mask.split()[3])

      result_image, result_mask, result_color_mask = self.image_resizing(result_image, result_mask, result_color_mask)

      return result_image, result_mask, result_color_mask

  def repeat(self,image, mask, color_mask, instance_mask, pattern_repeat_count):
    image_size = image.size
    numpy_image = np.array(image)
    original_opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # contour1 = self.find_contours(mask)
    contour1 = self.find_contours(mask, instance_mask)
    if (len(contour1) == 0):
        return None, None, None

    cls_id = contour1[0][1]
    contour = contour1[0][0]
    
    inner_region = self.extract_inner_region(original_opencv_image, contour)
    inner_image1 = Image.fromarray(inner_region)

    numpy_mask = np.array(mask)
    inner_mask1 = self.extract_inner_region(numpy_mask, contour)
    inner_mask1 = Image.fromarray(inner_mask1)

    numpy_color_mask = np.array(color_mask)
    inner_color_mask1 = self.extract_inner_region(numpy_color_mask, contour)
    inner_color_mask1 = inner_color_mask1[:, :, [2,1,0,3]]
    inner_color_mask1 = Image.fromarray(inner_color_mask1)

    new_image = copy.deepcopy(image)
    new_mask = copy.deepcopy(mask)
    new_color_mask = copy.deepcopy(color_mask)

    W, H = new_image.size
    obj_w, obj_h = inner_image1.size

    # 원래 객체 중심 계산
    x0, y0, w0, h0 = cv2.boundingRect(contour)
    cx0, cy0 = x0 + w0 // 2, y0 + h0 // 2
    existing_centers = [(cx0, cy0)]

    # 거리 기준
    min_dist_self = max(obj_w, obj_h) * 1.2     # 기존보다 완화
    min_dist_contour = int(W * 0.25)

    positions = []
    max_attempts = 300  # 총 시도 횟수 상향 (기존 100보다 여유 있게)

    # 패턴 개수 보장
    for i in range(pattern_repeat_count):
        found = False
        attempts = 0
        while not found and attempts < max_attempts:
            x = random.randint(0, W - obj_w)
            y = random.randint(0, H - obj_h)

            cx = x + obj_w // 2
            cy = y + obj_h // 2

            # 기존 객체 및 contour와의 거리 검사
            far_from_others = all(
                ((cx - px)**2 + (cy - py)**2)**0.5 > min_dist_self
                for px, py in positions
            )
            far_from_contour = all(
                ((cx - ex)**2 + (cy - ey)**2)**0.5 > min_dist_contour
                for ex, ey in existing_centers
            )

            if far_from_others and far_from_contour:
                positions.append((cx, cy))
                found = True
            attempts += 1

        # 300번 다 시도해도 못 찾으면 거리 조건을 완화해서라도 강제로 생성
        if not found:
            for _ in range(1000):  # 완화된 탐색
                x = random.randint(0, W - obj_w)
                y = random.randint(0, H - obj_h)
                cx = x + obj_w // 2
                cy = y + obj_h // 2
                if all(((cx - px)**2 + (cy - py)**2)**0.5 > (min_dist_self * 0.6) for px, py in positions):
                    positions.append((cx, cy))
                    found = True
                    break

        # 그래도 실패하면 그냥 랜덤으로라도 하나 붙이기
        if not found:
            x = random.randint(0, W - obj_w)
            y = random.randint(0, H - obj_h)
            cx, cy = x + obj_w // 2, y + obj_h // 2
            positions.append((cx, cy))

        # 실제 붙이기
        x_paste = positions[-1][0] - obj_w // 2
        y_paste = positions[-1][1] - obj_h // 2

        new_image.paste(inner_image1.convert("RGB"), (x_paste, y_paste), inner_image1.split()[3])
        new_mask.paste(inner_mask1.convert("RGB"), (x_paste, y_paste), inner_mask1.split()[3])
        new_color_mask.paste(inner_color_mask1.convert("RGB"), (x_paste, y_paste), inner_color_mask1.split()[3])

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