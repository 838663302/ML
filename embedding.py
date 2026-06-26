from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib

# 脚本文件所在目录（不依赖执行位置）
BASE_DIR = Path(__file__).parent.resolve()

mtcnn = MTCNN(image_size=160)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# @torch.inference_mode()
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
    
    total_time = 0.0
    count = 0

    for subdir in subdirs:
        sorted_files = sorted([f for f in subdir.iterdir() if f.is_file()])
        for file in sorted_files:
            start = time.time()
            embedding = get_embedding(file)
            elapsed = time.time() - start
            if embedding is not None:
                total_time += elapsed
                count += 1
                embeddings.append(embedding)
                labels.append(folderIndex)
                map_label[subdir.name] = folderIndex
        folderIndex += 1

    if count > 0:
        print(f"Processed {count} images, total {total_time:.3f}s, avg {total_time / count:.3f}s per image")

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
