from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import joblib
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# 脚本文件所在目录（不依赖执行位置）
BASE_DIR = Path(__file__).parent.resolve()

mtcnn = MTCNN(image_size=160)
mobilenet = models.mobilenet_v2(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# @torch.inference_mode()
def save_img(image_path, save_dir):
    path = Path(image_path)
    # 相对路径基于脚本所在目录解析
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        return None
    else:
        try:
            image = Image.open(path).convert('RGB')
            save_path = BASE_DIR / 'faceimg' / save_dir / path.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            mtcnn(image, save_path=str(save_path))
            return True
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

def loaddataset(dataset_path):
    path = Path(dataset_path)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        return None
    
    # 获取所有子文件夹（每个文件夹代表一个类别）
    subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} class(es): {[d.name for d in subdirs]}")
    

    for subdir in subdirs:
        sorted_files = sorted([f for f in subdir.iterdir() if f.is_file()])
        for file in sorted_files:
            save_img(file, subdir.name)

def load_path_labels(dir):
    paths = []
    labels = []
    class_map = {}
    subdirs = sorted([d for d in dir.iterdir() if d.is_dir()])

    for i, subdir in enumerate(subdirs):
        class_map[subdir.name] = i
        for file in subdir.iterdir():
            paths.append(file)
            labels.append(i)
    
    return paths, labels, class_map


class FaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class FaceModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = mobilenet
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.features[14:].parameters():
            param.requires_grad = True
        
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

def train_model():
    # 加载数据集
    image_paths, labels, class_map = load_path_labels(BASE_DIR / "faceimg")
    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, 
        random_state=42, stratify=labels)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # 创建数据加载器
    traindataset = FaceDataset(train_paths, train_labels, train_transforms)
    valdataset = FaceDataset(val_paths, val_labels, val_transforms)
    train_loader = DataLoader(traindataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(valdataset, batch_size=32, shuffle=False)
    model = FaceModel(num_classes=len(class_map))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    # 差异化学习率：backbone用小学习率，分类器用大学习率
    backbone_params = [p for name, p in model.named_parameters() if 'backbone.features' in name and p.requires_grad]
    classifier_params = [p for name, p in model.named_parameters() if 'classifier' in name]
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 0.0001},
        {'params': classifier_params, 'lr': 0.001}
    ])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)
    num_epochs = 25
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        lossvalue = 0
        acc = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            lossvalue += loss.item()
            acc += (outputs.argmax(dim=1) == labels.to(device)).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{num_epochs}, train Loss: {lossvalue/len(train_loader)} Accuracy: {acc/total}")
        scheduler.step()

        model.eval()
        test_acc = 0
        test_total = 0
        with torch.no_grad():
            acc = 0
            for images, labels in val_loader:
                outputs = model(images.to(device))
                test_acc += (outputs.argmax(dim=1) == labels.to(device)).sum().item()
                test_total += labels.size(0)
        print(f"Epoch {epoch+1}/{num_epochs}, test Loss: {lossvalue/len(val_loader)} Accuracy: {test_acc/test_total:.4f}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            predicted = outputs.argmax(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    # 索引 -> 类别名
    idx_to_class = {v: k for k, v in class_map.items()}
    y_true = [idx_to_class[i] for i in all_labels]
    y_pred = [idx_to_class[i] for i in all_preds]

    print("\n分类报告:")
    print(classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    print(cm)

if __name__ == '__main__':
    # loaddataset('dataset')
    train_model()
