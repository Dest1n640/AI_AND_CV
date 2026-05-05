import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path 
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

dataset_path = Path("./roads")

class RoadsDataset(Dataset):
  def __init__(self, dataset_path):
    super().__init__()
    self.images_path = dataset_path / "images"
    self.masks_path = dataset_path / "masks"
    self.images = list(self.images_path.glob("*.png"))
    self.mask = list(self.masks_path.glob("*.png"))
    self.len = len(list(self.images_path.glob("*.png")))

  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    #TODO понять почему без деления на 255 получается плохо все 
    image = Image.open(self.images[index]).convert("RGB")
    image = np.array(image, dtype="f4") / 255.
    mask = Image.open(self.mask[index]).convert("L")
    mask = np.array(mask, dtype="f4")
    mask = (mask == 82).astype("f4")
    mask = np.expand_dims(mask, axis=0) #1, H, W
    if np.random.rand() > 0.5:
      image = np.flip(image, axis=1).copy()
      mask = np.flip(mask, axis=2).copy()
    image = torch.from_numpy(image.transpose(2, 0, 1)) # C, H, W
    mask = torch.from_numpy(mask)
    return image, mask

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1, 1)),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),

      nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1, 1)),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
    )
  def forward(self, x):
    return self.conv(x)
  
class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
    super().__init__()
    self.downscale = nn.ModuleList()
    self.upscale = nn.ModuleList()
    self.pool = nn.MaxPool2d(2, 2)
    for n in features:
      self.downscale.append(DoubleConv(in_channels, n))
      in_channels = n
    
    for n in reversed(features):
      self.upscale.append(nn.ConvTranspose2d(n * 2, n, 2, 2))
      self.upscale.append(DoubleConv(n * 2, n))

    self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
    self.result = nn.Conv2d(features[0], out_channels, 1)
  #TODO разобрать построение архитектуры всей модели 
  def forward(self, x):
    skips = []

    for ds in self.downscale:
      x = ds(x)
      skips.append(x)
      x = self.pool(x)

    x = self.bottleneck(x)

    skips = skips[::-1]
    for idx in range(0, len(self.upscale, 2)):
      x = self.upscale[idx](x)
      skip = skips[idx // 2]
      cx = torch.cat((skip, x), dim=1)
      x = self.upscale[idx + 1](cx)
    return self.result(x)

#TODO разобраться с loss 
class DiceLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, pred, target):
    pred_sig = torch.sigmoid(pred)
    p_area = pred.sig.view(-1)
    t_area = target.view(-1)
    intersection = (p_area * t_area).sum()
    return 1 - (2 * intersection + 1) / (pred_sig.sum() + t_area.sum() + 1)

roads_dataset = RoadsDataset(dataset_path)
img, mask = roads_dataset[0]

unet = UNet()
