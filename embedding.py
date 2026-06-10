from pathlib import Path
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import joblib

# 脚本文件所在目录（不依赖执行位置）
BASE_DIR = Path(__file__).parent.resolve()

mtcnn = MTCNN(image_size=160)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(image_path):
    path = Path(image_path)
    # 相对路径基于脚本所在目录解析
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        return None
    else:
        try:
            image = Image.open(path).convert('RGB')
            img_cropped = mtcnn(image)
            if img_cropped is None:
                print(f"No face detected in {image_path}")
                return None
            with torch.no_grad():
                embedding = resnet(img_cropped.unsqueeze(0)).numpy().flatten()
                return embedding
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None


def loaddataset(dataset_path):
    path = Path(dataset_path)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    if not path.exists():
        return None
    
    map_label = {}
    labels = []
    embeddings = []
    folderIndex = 1
    
    # 获取所有子文件夹（每个文件夹代表一个类别）
    subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} class(es): {[d.name for d in subdirs]}")
    
    for subdir in subdirs:
        sorted_files = sorted([f for f in subdir.iterdir() if f.is_file()])
        for file in sorted_files:
            embedding = get_embedding(file)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(folderIndex)
                map_label[subdir.name] = folderIndex
        folderIndex += 1
    
    return np.array(embeddings), np.array(labels), map_label

def train():
    result = loaddataset("dataset")

    if result is None:
        print("Dataset not found. Exiting.")
        return
    embeddings, labels, map_label = result

    # 保存数据集到文件
    np.save("embeddings.npy", embeddings)
    np.save("labels.npy", labels)
    joblib.dump(map_label, "map_label.pkl")

    if len(embeddings) == 0:
        print("No valid embeddings extracted. Exiting.")
        return
    num_classes = len(map_label)
    if num_classes < 2 or len(embeddings) < num_classes * 5:
        print(f"Warning: Not enough data for training.")
        print(f"  - Need at least 2 classes (currently {num_classes})")
        print(f"  - Need more samples (currently {len(embeddings)})")
        print("Please add more folders and images to the dataset.")
        return

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 绘制 ROC 曲线（多分类 One-vs-Rest）
    y_proba = model.predict_proba(X_test)
    classes = model.classes_
    print("Classes:", classes)
    y_test_bin = label_binarize(y_test, classes=classes)
    print("y_test_bin shape:", y_test_bin)

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        name = [k for k, v in map_label.items() if v == cls][0]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.show()
    print("ROC 曲线已保存到 roc_curve.png")

    # embedding = get_embedding("liuyifei/1.png")
    # if embedding is None:
    #     print("Failed to get embedding for prediction image.")
    #     return
    # embedding = embedding.reshape(1, -1)
    # probabilities = model.predict_proba(embedding)[0]
    # max_prob = np.max(probabilities)
    
    # if max_prob < 0.9:
    #     print(f"预测类别: unknown (最大概率 {max_prob:.2%} < 90%)")
    # else:
    #     prediction = model.predict(embedding)[0]
    #     print(f"预测类别: {prediction} (概率 {max_prob:.2%})")
    
    # print(f"各类别概率: {probabilities}")


if __name__ == "__main__":
    train()
