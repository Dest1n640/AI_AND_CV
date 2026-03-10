import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

save_path = Path(__file__).parent

# device = torch.device("cuda"
#                       if torch.cuda.is_available() else 'cpu')

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"{device}")

torch.manual_seed(42)

transforms = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5, ))
])

batch_size = 64

train_dataset = datasets.FashionMNIST(root="./tmp",
                                      train=True,
                                      download=True,
                                      transform=transforms)

test_dataset = datasets.FashionMNIST(root="./tmp",
                                      train=False,
                                      download=True,
                                      transform=transforms)

train_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle = True)
test_loader = DataLoader(train_dataset,
                          batch_size = batch_size,
                          shuffle = False)

print(f"{len(train_loader)}, {len(test_loader)}")

plt.figure()
for i in range(9):
  image, label = train_dataset[i]
  image = image.numpy().transpose(1, 2, 0)
  plt.subplot(3, 3, i + 1)
  plt.title(f"{label=}")
  plt.imshow(image)
plt.tight_layout()
plt.show()

class FashionCNN(nn.Module):
  def __init__(self):
    super(FashionCNN, self).__init__()
    #1
    self.conv1 = nn.Conv2d(in_channels=1,
                           out_channels=32,
                           kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, 2) #28, 28 -> 14, 14

    #2
    self.conv2 = nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2, 2) # 14, 14 -> 7, 7

    self.flatten = nn.Flatten()
    
    self.fc1 = nn.Linear(64 * 7 * 7, 128)
    self.relu3 = nn.ReLU()
    self.dropout = nn.Dropout(0.5)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
    x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
    x = self.flatten(x)

    x = self.relu3(self.fc1(x))
#     x = self.dropout(x)
    x = self.fc2(x)
    return x
  
model = FashionCNN()
total_params = sum(p.numel() for p in model.parameters()) 
print(f"params = {total_params}")

criteion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 10
train_loss = []
train_acc = []

model_path = save_path / "tmp/model.pth"
if not model_path.exists():

  for epoch in range(num_epochs):
    model.train()
    run_loss = 0.0
    total = 0
    correct = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
  #     images, labels = (images.to(device), labels.to(device))
      optimizer.zero_grad()
      outputs = model(images)
      loss = criteion(outputs, labels)
      loss.backward()
      optimizer.step()
      run_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
    scheduler.step()
    epoch_loss = run_loss / len(train_loader)
    epoch_acc = 100 * (correct / total)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    print(f"Epoch {epoch}, {epoch_loss:=.3f}, {epoch_acc:=.3f}")
  torch.save(model.state_dict(), model_path)
  plt.figure()
  plt.subplot(121)
  plt.title("Loss")
  plt.plot(train_loss)
  plt.subplot(122)
  plt.plot(train_acc)
  plt.show()

else:
  model.load_state_dict(torch.load(model_path))

model.eval()
it = iter(test_loader)
images, labels = next(it)
image = images[0].unsqueeze(0)
# image = image.to(device)
with torch.no_grad():
  output = model(image)
  _, predicted = torch.max(output, 1)

classes = ['T_shirt', "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"True - {classes[labels[0]]}")
print(f"Pred - {classes[predicted.cpu().item()]}")
