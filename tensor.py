import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchsummary as summary
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.nn as nn
from torchvision import models
mobilenet = models.mobilenet_v2(pretrained=True)
resnet = InceptionResnetV1(pretrained='vggface2')
print(mobilenet.features[0][0])
for name, module in mobilenet.named_children():
    print(name,"----", module)
# summary.summary(mobilenet, (3, 224, 224), -1, 'cpu')
# DataLoader()
# mtcnn = MTCNN(image_size=160)
# img = mtcnn(Image.open("./1.png").convert('RGB'), save_path="./1_cropped.png")
# print(img)

# InceptionResnetV1 = InceptionResnetV1(pretrained='vggface2').eval()
# print(InceptionResnetV1(img.unsqueeze(0)))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
# summary.summary(InceptionResnetV1(pretrained='vggface2'), (3, 160, 160), -1, 'cpu')
# dir = sorted([d for d in Path(__file__).parent.resolve().iterdir() if d.is_dir()])
# print(dir)
# print(device)
# for name , param in InceptionResnetV1(pretrained='vggface2').named_modules():
#     print(name, param)

# x = torch.randint(0,10,(2,2,2))
# print(x.long())
# print(torch.tensor(1).shape)
# print(x.device)
# y = torch.Tensor([2,3])
# print(y.dtype)
# print(y)
# print(torch.tensor([2,3]).dtype)
# print(torch.linspace(0,5,7))
# z = torch.full((2,3), 6)
# print(z)
# print(torch.full_like(z, 7))
# print(torch.empty_like(z))
# torch.LongTensor([1,2,3])
# print(torch.tensor(np.random.rand(2,3)))
# print(torch.tensor(np.random.randint(0,10,(2,3))).float().exp())
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
#         self.sigmoid1 = nn.Sigmoid()
#         self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.sigmoid2 = nn.Sigmoid()
#         self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(400, 120)
#         self.sigmoid3 = nn.Sigmoid()
#         self.fc2 = nn.Linear(120, 84)
#         self.sigmoid4 = nn.Sigmoid()
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.sigmoid1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.sigmoid2(x)
#         x = self.pool2(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.sigmoid3(x)
#         x = self.fc2(x)
#         x = self.sigmoid4(x)
#         x = self.fc3(x)
#         return x

# model = SimpleCNN()

# x = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# for name, module in model.named_children():
#     print(name, module)
#     x = module(x)
#     print(x.shape)

