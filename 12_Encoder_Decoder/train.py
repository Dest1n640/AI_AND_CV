import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageDraw2, ImageFont
import numpy as np
import matplotlib.pyplot as plt


class ImageDataset(Dataset):
  def __init__(self, n=200, size=256):
    super().__init__()
    self.n = n
    self.size = size
    self.transforms = transforms.Compose([
      transforms.ToTensor()
    ])

  def __len__(self):
    return self.n
  
  def __getitem__(self, index):
    image = Image.new("L", (self.size, self.size), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = "ABC"
    # x = np.random.randint(10, self.size - 40) 
    # y = np.random.randint(10, self.size - 40)
    x = 30
    y = 30
    draw.text((x, y), text, fill=0, font=font)
    tensor = self.transforms(image)
    return tensor, tensor
  
class Encoder(nn.Module):
  def __init__(self, latent_size=512):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, 32, 3, stride=2, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 64, 3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.Conv2d(64, 128, 3, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.Conv2d(128, 256, 3, stride=2, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU()
    )

    self.bottleneck = nn.Linear(256 * 16 * 16, latent_size)

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    return x

class Decoder(nn.Module):
  def __init__(self, latent_size=512):
    super().__init__()
    self.bottleneck = nn.Linear(latent_size, 256 * 16 * 16)
    self.features = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(), 

      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.bottleneck(x)
    x = x.view(x.size(0), 256, 16, 16)
    x = self.features(x)
    return x

if __name__ == "__main__":
  encoder = Encoder()
  decoder = Decoder()
  encoder_params = sum(p.numel() for p in encoder.parameters())
  decoder_params = sum(p.numel() for p in decoder.parameters())
  print(encoder_params)
  print(decoder_params)

  data = ImageDataset()
  data_loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=2)

  device = torch.device("mps" if torch.mps.is_available else "cpu")
  encoder.to(device)
  decoder.to(device)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(list(encoder.parameters())+
                        list(decoder.parameters()))
  encoder.train()
  decoder.train()
  epochs = 10

  for epoch in range(epochs):
    epoch_loss = 0.0
    for imgs, _ in data_loader:
      imgs = imgs.to(device)
      optimizer.zero_grad()
      latent = encoder(imgs)
      output = decoder(latent)
      loss = criterion(output, imgs)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch - {epoch}\n avg_loss - {avg_loss=:.2f}")

  torch.save(encoder.state_dict(), "encoder.pth")
  torch.save(decoder.state_dict(), "decoder.pth")

# plt.imshow(data[1][0][0])
# plt.show()

