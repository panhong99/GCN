from __future__ import print_function
import cv2
import torch.utils.data as data
import os
from PIL import Image,ImageOps
import numpy as np
import scipy.ndimage as ndi
import random
from utils import preprocess

class VOCSegmentation(data.Dataset):
  CLASSES = [
      'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
      'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
      'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
      'tv/monitor'
  ]

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None,process=None,process_value=None,overlap_percentage=None,pattern_repeat_count=None):
    self.root = root
    _voc_root = os.path.join(self.root, 'VOC2012')
    _list_dir = os.path.join(_voc_root, 'list')
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size
    self.process = process
    self.process_value = process_value
    self.overlap_percentage = overlap_percentage
    self.pattern_repeat_count = pattern_repeat_count
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
    with open(_list_f, 'r') as lines:
      for line in lines:
        _image = _voc_root + line.split()[0]
        _mask = _voc_root + line.split()[1]
        assert os.path.isfile(_image)
        assert os.path.isfile(_mask)
        self.images.append(_image)
        self.masks.append(_mask)

  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')

    _target = Image.open(self.masks[index])
    if self.process != None:
      _target= _target.convert('L')


    # add image process for test
    if self.process == 'zoom':
      _img, _target = self.zoom_center(_img, _target, self.process_value)

    elif self.process == 'overlap' and index < len(self.images) - 1:

      next_img=Image.open(self.images[index+1]).convert('RGB')
      next_target=Image.open(self.masks[index+1])
      _img, _target = self.overlap(_img, _target,next_img,next_target,self.overlap_percentage)
      if _img==None:
        return None,None


    elif self.process == 'repeat':
      _img, _target = self.repeat(_img, _target,self.pattern_repeat_count)
      if _img==None:
        return None,None



    #if self.process == None:
    _img, _target = preprocess(_img, _target,
                               flip=True if self.train else False,
                               scale=(0.5, 2.0) if self.train else None,
                               crop=(self.crop_size, self.crop_size))
    if self.transform is not None:
      _img = self.transform(_img)

    #print("self.target_transform ",self.target_transform )
    if self.target_transform is not None:
      _target = _target.unsqueeze(0)
      _target = self.target_transform(_target)



    return _img, _target

  def __len__(self):
    return len(self.images)

  def download(self):
    raise NotImplementedError('Automatic download not yet implemented.')

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

  def overlap(self, image, mask, next_image, next_mask, overlap_percentage=0.5):
    numpy_image = np.array(image)
    original_opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    next_numpy_image = np.array(next_image)
    next_original_opencv_image = cv2.cvtColor(next_numpy_image, cv2.COLOR_RGB2BGR)

    contour1 = self.find_contours(mask)
    if len(contour1) != 1:
      return None,None  # If contours are not exactly one, skip processing

    inner_region1 = self.extract_inner_region(original_opencv_image, contour1[0])
    inner_image1 = Image.fromarray(cv2.cvtColor(inner_region1, cv2.COLOR_BGR2RGB))

    contour2 = self.find_contours(next_mask)
    if len(contour2) != 1:
      return None,None  # If contours are not exactly one, skip processing

    inner_region2 = self.extract_inner_region(next_original_opencv_image, contour2[0])
    inner_image2 = Image.fromarray(cv2.cvtColor(inner_region2, cv2.COLOR_BGR2RGB))


    inner_image1_np = np.array(inner_image1)
    inner_image2_np = np.array(inner_image2)
    inner_mask1_np = np.array(mask)
    inner_mask2_np = np.array(next_mask)

    # Calculate overlap width
    overlap_width = int(inner_image1_np.shape[1] * overlap_percentage)

    # Create a canvas to overlay both images
    canvas_width = inner_image1_np.shape[1] + inner_image2_np.shape[1] - overlap_width
    canvas_height = max(inner_image1_np.shape[0], inner_image2_np.shape[0])
    canvas_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Place the first image and mask on the canvas
    canvas_image[:inner_image1_np.shape[0], :inner_image1_np.shape[1]] = inner_image1_np
    canvas_mask[:inner_mask1_np.shape[0], :inner_mask1_np.shape[1]] = inner_mask1_np

    # Overlay the second image and mask on the canvas with overlap
    start_x = inner_image1_np.shape[1] - overlap_width
    for i in range(inner_image2_np.shape[0]):
      for j in range(inner_image2_np.shape[1]):
        if canvas_mask[i, start_x + j] == 0:
          canvas_image[i, start_x + j] = inner_image2_np[i, j]
          canvas_mask[i, start_x + j] = inner_mask2_np[i, j]

    # Convert the canvas back to an image and mask
    result_image = Image.fromarray(cv2.cvtColor(canvas_image, cv2.COLOR_BGR2RGB))
    result_mask = Image.fromarray(canvas_mask)

    return result_image, result_mask

  def repeat(self,image, mask, pattern_repeat_count):
    image_size = image.size
    numpy_image = np.array(image)
    original_opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    contour1 = self.find_contours(mask)
    if len(contour1) != 1:
      return None, None  # If contours are not exactly one, skip processing

    inner_region1 = self.extract_inner_region(original_opencv_image, contour1[0])
    inner_image1 = Image.fromarray(cv2.cvtColor(inner_region1, cv2.COLOR_BGR2RGB))
    inner_image1_resize = inner_image1.resize(
      (image_size[0], image_size[1])
    )

    numpy_mask = np.array(mask)
    original_opencv_mask = cv2.cvtColor(numpy_mask, cv2.COLOR_GRAY2BGR)
    inner_mask1 = self.extract_inner_region(original_opencv_mask, contour1[0])
    inner_mask1_resize = Image.fromarray(cv2.cvtColor(inner_mask1, cv2.COLOR_BGR2GRAY)).resize(
      (image_size[0], image_size[1])
    )

    # Create empty new images and masks of the same size as the original image
    new_image_size = (image_size[0] * pattern_repeat_count, image_size[1] * pattern_repeat_count)
    new_image = Image.new('RGB', new_image_size)
    new_mask = Image.new('L', new_image_size)

    # Paste the original image and mask in a grid pattern
    for i in range(pattern_repeat_count):
      for j in range(pattern_repeat_count):
        new_image.paste(inner_image1_resize, (i * image_size[0], j * image_size[1]))
        new_mask.paste(inner_mask1_resize, (i * image_size[0], j * image_size[1]))

    # Resize the final images to the original image size
    final_image = new_image.resize(image_size)
    final_mask = new_mask.resize(image_size)

    return final_image, final_mask




  def zoom_center(self, image, mask, zoom_factor):
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

    elif zoom_factor < 1:
      # Zoom out
      new_width = int(width * zoom_factor)
      new_height = int(height * zoom_factor)

      # Resize the image and mask to the new dimensions
      resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
      resized_mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)

      # Create a new black image and paste the resized image and mask in the center
      new_image = Image.new('RGB', (width, height), (255, 255, 255))
      new_mask = Image.new('L', (width, height), 255)

      new_image.paste(resized_image, ((width - new_width) // 2, (height - new_height) // 2))
      new_mask.paste(resized_mask, ((width - new_width) // 2, (height - new_height) // 2))

      image = new_image
      mask = new_mask

    return image, mask

