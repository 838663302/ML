import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
from embedding import get_embedding

# Configuration
MODEL_FILE = "./model.pkl"
LABEL_MAP_FILE = "./labels.pkl"

# Initialize facenet models
print("Loading models...")
mtcnn = MTCNN(image_size=160, margin=10)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load classifier
model = joblib.load(MODEL_FILE)
label_map = joblib.load(LABEL_MAP_FILE)
reverse_label_map = {v: k for k, v in label_map.items()}

def predict(image_path):
    embedding = get_embedding(image_path)
    if embedding is None:
        return None, "No face detected", None

    # Predict
    pred_label = model.predict(embedding)[0]
    pred_proba = model.predict_proba(embedding)[0]
    
    person_name = reverse_label_map[pred_label]
    confidence = pred_proba[pred_label]
    
    if confidence < 0.8:
        return None, "置信度不足 80%，无法识别", pred_proba
    
    return person_name, confidence, pred_proba

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print(f"Available classes: {list(label_map.keys())}")
        sys.exit(1)
    
    image_path = sys.argv[1]
    person, confidence, proba = predict(image_path)
    
    if person is None:
        print(f"Error: {confidence}")
    else:
        print(f"Prediction: {person}")
        print(f"Confidence: {confidence:.2%}")
        
        # Show all probabilities
        print("\nAll probabilities:")
        for i, p in enumerate(proba):
            name = reverse_label_map[i]
            print(f"  {name}: {p:.2%}")
