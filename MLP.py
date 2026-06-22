import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

Root_path = Path(__file__).parent.resolve()
# ==================== 数据集 ====================
class FaceDataset(Dataset):
    SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []       # [(图片路径, 标签索引), ...]
        self.classes = []       # 类别名列表
        self.class_to_idx = {}  # 类别名 -> 索引


        subdirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        for idx, subdir in enumerate(subdirs):
            self.classes.append(subdir.name)
            self.class_to_idx[subdir.name] = idx
            for img in sorted([d for d in subdir.iterdir() if d.is_file()]):
                if img.suffix.lower() in FaceDataset.SUPPORTED_EXT:
                    self.samples.append((str(img), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ==================== 迁移学习模型 ====================
class TransferModel(nn.Module):
    """
    基于 FaceNet (InceptionResnetV1) 预训练权重的迁移学习模型
    冻结 backbone，只训练全连接分类头
    """
    def __init__(self, num_classes, hidden_dim=256, dropout=0.3):
        super(TransferModel, self).__init__()
        # 加载预训练 FaceNet backbone（输出 512 维特征）
        self.backbone = InceptionResnetV1(pretrained='vggface2')

        # 冻结 backbone 所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 全连接分类头（不使用 Sequential，逐层定义）
        self.fc1 = nn.Linear(512, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """对全连接层使用 Kaiming 初始化"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # backbone 提取 512 维特征
        x = self.backbone(x)
        # 分类头
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


# ==================== 训练函数 ====================
def train(dataset_dir="dataset", epochs=30, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # FaceNet 标准预处理: 160x160 + 归一化
    data_transforms = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    data_dir = Path(Root_path / dataset_dir).resolve()
    if not data_dir.exists():
        print("数据集目录不存在，请检查 dataset/ 文件夹。")
        return
    if not data_dir.is_dir():
        print("数据集目录不存在，请检查 dataset/ 文件夹。")
        return

    # 加载数据集
    full_dataset = FaceDataset(data_dir, transform=data_transforms)
    if len(full_dataset) == 0:
        print("未找到任何图片，请检查 dataset/ 文件夹。")
        return
    num_classes = len(full_dataset.classes)
    print(f"共加载 {len(full_dataset)} 张图片, {num_classes} 个类别: {full_dataset.classes}")
    print(full_dataset.samples)
    

    # 划分训练集和测试集（按样本索引划分）
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)), test_size=0.2, random_state=42,
        stratify=[full_dataset.samples[i][1] for i in range(len(full_dataset))]
    )
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    test_subset = torch.utils.data.Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    print(f"训练集: {len(train_subset)} 张 | 测试集: {len(test_subset)} 张")

    # 初始化模型
    model = TransferModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # 只优化分类头的参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练循环
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = correct / total

        # 在测试集上评估
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = test_correct / test_total

        print(f"轮次 [{epoch+1}/{epochs}] | "
              f"损失: {total_loss/len(train_loader):.4f} | "
              f"训练准确率: {train_acc:.4f} | 测试准确率: {test_acc:.4f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "mlp_model.pth")
            print(f"  -> 保存最佳模型 (测试准确率: {best_acc:.4f})")

    print(f"\n训练完成！最佳测试准确率: {best_acc:.4f}")

    # 加载最佳模型，输出分类报告
    model.load_state_dict(torch.load("mlp_model.pth", weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    # 索引 -> 类别名
    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    y_true = [idx_to_class[i] for i in all_labels]
    y_pred = [idx_to_class[i] for i in all_preds]

    print("\n分类报告:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # 保存类别映射
    joblib.dump(full_dataset.class_to_idx, "map_label.pkl")
    print("模型已保存到 mlp_model.pth, 类别映射已保存到 map_label.pkl")


if __name__ == "__main__":
    train()
