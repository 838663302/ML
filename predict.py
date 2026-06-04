import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib

# Configuration
MODEL_FILE = "./classifier.pkl"
LABEL_MAP_FILE = "./label_map.pkl"

# Initialize facenet models
print("Loading models...")
mtcnn = MTCNN(image_size=160, margin=10)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load classifier
clf = joblib.load(MODEL_FILE)
label_map = joblib.load(LABEL_MAP_FILE)
reverse_label_map = {v: k for k, v in label_map.items()}

def predict(image_path):
    """Predict the class of a face image."""
    img = Image.open(image_path).convert('RGB')
    
    # Detect and crop face
    img_cropped = mtcnn(img)
    if img_cropped is None:
        return None, "No face detected"
    
    # Get embedding
    img_cropped = img_cropped.unsqueeze(0)
    with torch.no_grad():
        embedding = resnet(img_cropped).numpy()
    
    # Predict
    pred_label = clf.predict(embedding)[0]
    pred_proba = clf.predict_proba(embedding)[0]
    
    person_name = reverse_label_map[pred_label]
    confidence = pred_proba[pred_label]
    
    return person_name, confidence

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print(f"Available classes: {list(label_map.keys())}")
        sys.exit(1)
    
    image_path = sys.argv[1]
    person, confidence = predict(image_path)
    
    if person is None:
        print(f"Error: {confidence}")
    else:
        print(f"Prediction: {person}")
        print(f"Confidence: {confidence:.2%}")
        
        # Show all probabilities
        print("\nAll probabilities:")
        img = Image.open(image_path).convert('RGB')
        img_cropped = mtcnn(img).unsqueeze(0)
        with torch.no_grad():
            embedding = resnet(img_cropped).numpy()
        proba = clf.predict_proba(embedding)[0]
        for i, p in enumerate(proba):
            name = reverse_label_map[i]
            print(f"  {name}: {p:.2%}")
