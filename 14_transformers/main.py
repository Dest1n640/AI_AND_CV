import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class ViT(nn.Module):
  def __int__(self, image_size=32, patch_size=4, channels=3,
              num_classes=10, embed_size=192, depth=6, 
              num_heads=3, mlp_coeff=4, drop_rate=0.1):
    super().__init__()
    assert image_size % patch_size == 0, "Wrong pathch size"
    num_patches = (image_size // patch_size) ** 2
    self.patch_embed = nn.Conv2d(channels, embed_size, patch_size, patch_size)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_size))
    self.pos_drop = nn.Dropout(drop_rate)
    encoder = nn.TransformerEncoderLayer(d_model=embed_size, n_head=num_heads,
                                         dim_feedforward=int(embed_size * mlp_coeff),
                                         dropout=drop_rate, activation='gelu')
    self.blocks = nn.TransformerEncoder(encoder, num_layers=depth)
    self.norm = nn.LayerNorm(embed_size)
    self.haed = nn.Sequential(
      nn.Linear(embed_size, num_classes)
    )
  def forward(self, x):
    size = x.shape[0]
    x = self.patch_embed(x).flatten(2).transpose(1, 2)
    x = torch.cat([self.cls_token.expand(size, -1, -1), x], dim = 1)
    x = x + self.pos_embed[:, :x.size(1), :]
    x = self.pos_drop(x)
    x = self.blocks(x)
    return self.haed(self.norm(x[:, 0]))



def choose_device():
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  return device


if __name__ == "__main__":
  device = choose_device()
  print(device)

  train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.247, 0.2435, 0.2616))
  ])

  test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.247, 0.2435, 0.2616))
  ])

  train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                   download=True, transform=train_transforms)
  
  test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                  download=True, transform=test_transforms)

  train_loader = DataLoader(train_dataset, batch_size=128, 
                            shuffle=True, num_workers=2)
  
  test_loader = DataLoader(test_dataset, batch_size=128, 
                           shuffle=False, num_workers=2)
  
  model = ViT()
  print(sum(p.numel() for p in model.parameters()))
  
  image, label = train_dataset[0]
  image = image.permute(1, 2, 0)
  plt.figure(figsize=(5, 5))
  plt.imshow(image)
  plt.title(f"Label: {label}")
  plt.show()
