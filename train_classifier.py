import os
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Configuration
DATASET_DIR = "./dataset"
EMBEDDINGS_FILE = "./embeddings.npy"
LABELS_FILE = "./labels.npy"
MODEL_FILE = "./classifier.pkl"
LABEL_MAP_FILE = "./label_map.pkl"

# Initialize facenet models
print("Loading models...")
mtcnn = MTCNN(image_size=160, margin=10)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_embedding(image_path):
    """Extract face embedding from a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        # Detect and crop face
        img_cropped = mtcnn(img)
        if img_cropped is None:
            print(f"  Warning: No face detected in {image_path}")
            return None
        # Add batch dimension
        img_cropped = img_cropped.unsqueeze(0)
        # Get embedding
        with torch.no_grad():
            embedding = resnet(img_cropped)
        return embedding.numpy().flatten()
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return None

def load_dataset(dataset_dir):
    """Load all images from dataset directory. Each subfolder is a class."""
    embeddings = []
    labels = []
    label_map = {}  # folder_name -> numeric_label
    label_idx = 0
    
    # Get all subfolders (each represents a person/class)
    class_folders = sorted([
        f for f in os.listdir(dataset_dir) 
        if os.path.isdir(os.path.join(dataset_dir, f))
    ])
    
    if len(class_folders) == 0:
        print("Error: No subfolders found in dataset directory!")
        return None, None, None
    
    print(f"\nFound {len(class_folders)} class(es): {class_folders}")
    
    for folder_name in class_folders:
        folder_path = os.path.join(dataset_dir, folder_name)
        label_map[folder_name] = label_idx
        
        # Get all image files
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        print(f"\nProcessing '{folder_name}' ({len(image_files)} images)...")
        
        for img_file in sorted(image_files):
            img_path = os.path.join(folder_path, img_file)
            embedding = extract_embedding(img_path)
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(label_idx)
        
        label_idx += 1
    
    return np.array(embeddings), np.array(labels), label_map

def main():
    # Step 1: Load dataset and extract embeddings
    print("=" * 50)
    print("Step 1: Extracting face embeddings")
    print("=" * 50)
    
    embeddings, labels, label_map = load_dataset(DATASET_DIR)
    
    if embeddings is None or len(embeddings) == 0:
        print("No valid embeddings extracted. Exiting.")
        return
    
    print(f"\nTotal embeddings: {len(embeddings)}, shape: {embeddings.shape}")
    print(f"Label map: {label_map}")
    
    # Step 2: Save embeddings
    print("\n" + "=" * 50)
    print("Step 2: Saving embeddings")
    print("=" * 50)
    
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(LABELS_FILE, labels)
    joblib.dump(label_map, LABEL_MAP_FILE)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")
    print(f"Labels saved to {LABELS_FILE}")
    print(f"Label map saved to {LABEL_MAP_FILE}")
    
    # Step 3: Train logistic regression classifier
    print("\n" + "=" * 50)
    print("Step 3: Training Logistic Regression Classifier")
    print("=" * 50)
    
    num_classes = len(np.unique(labels))
    
    if len(embeddings) < 5 or num_classes < 2:
        print(f"\nWarning: Not enough data for training.")
        print(f"  - Need at least 2 classes (currently {num_classes})")
        print(f"  - Need more samples (currently {len(embeddings)})")
        print("Please add more folders and images to the dataset.")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    clf = LogisticRegression(
        max_iter=1000, 
        random_state=42,
        multi_class='multinomial' if num_classes > 2 else 'auto'//softmax回归
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    
    # Create reverse label map for display
    reverse_label_map = {v: k for k, v in label_map.items()}
    target_names = [reverse_label_map[i] for i in range(num_classes)]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Step 4: Save classifier
    print("=" * 50)
    print("Step 4: Saving classifier")
    print("=" * 50)
    
    joblib.dump(clf, MODEL_FILE)
    print(f"Classifier saved to {MODEL_FILE}")
    
    print("\n" + "=" * 50)
    print("Done! You can now use the saved model for prediction.")
    print("=" * 50)

if __name__ == "__main__":
    main()
