import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.models as models
import timm

model = timm.create_model("vit_small_patch16_224" , pretrained=False)
model.train()

