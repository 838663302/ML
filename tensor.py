import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchsummary as summary
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# DataLoader()
mtcnn = MTCNN(image_size=160)
img = mtcnn(Image.open("./1.png").convert('RGB'), save_path="./1_cropped.png")
print(img)

InceptionResnetV1 = InceptionResnetV1(pretrained='vggface2').eval()
print(InceptionResnetV1(img.unsqueeze(0)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# summary.summary(InceptionResnetV1(pretrained='vggface2'), (3, 160, 160), -1, 'cpu')
# dir = sorted([d for d in Path(__file__).parent.resolve().iterdir() if d.is_dir()])
# print(dir)
print(device)

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
