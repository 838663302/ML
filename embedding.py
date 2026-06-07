from pathlib import Path
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
    folderIndex = 0
    
    # 获取所有子文件夹（每个文件夹代表一个类别）
    subdirs = sorted([d for d in path.iterdir() if d.is_dir()])
    print(f"Found {len(subdirs)} class(es): {[d.name for d in subdirs]}")
    
    for subdir in subdirs:
        sorted_files = sorted([f for f in subdir.iterdir() if f.is_file()])
        for file in sorted_files:
            embedding = get_embedding((path / file).resolve())
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(folderIndex)
                map_label[subdir.name] = folderIndex
        folderIndex += 1
    
    return np.array(embeddings), np.array(labels), map_label

def train():
    embeddings, labels, map_label = loaddataset("dataset")

    if embeddings is None or len(embeddings) == 0:
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
    model = LogisticRegression(max_iter=1000, multi_class='multinomial' if num_classes > 2 else 'auto', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train()
