import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import joblib


def train_model(embeddings, labels, map_label):
    """
    训练模型并生成 ROC 曲线
    
    Args:
        embeddings: 特征向量数组
        labels: 标签数组
        map_label: 类别映射字典
    """
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


if __name__ == "__main__":
    # 从保存的文件加载数据并训练
    embeddings = np.load("embeddings.npy")
    labels = np.load("labels.npy")
    map_label = joblib.load("map_label.pkl")
    train_model(embeddings, labels, map_label)
