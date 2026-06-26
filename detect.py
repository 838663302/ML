import cv2
import torch
import ctypes
import time
import tkinter as tk
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import joblib
from MLP import TransferModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, keep_all=True, device=device)

# 加载标签映射
label_map = joblib.load("map_label.pkl")
reverse_label_map = {v: k for k, v in label_map.items()}

# 加载模型
model = TransferModel(num_classes=len(label_map)).to(device)
model.load_state_dict(torch.load("mlp_model.pth", weights_only=True))
model.eval()


def show_countdown_and_lock():
    """显示2秒倒计时弹框，然后锁屏"""
    root = tk.Tk()
    root.title("警告")
    root.geometry("300x150")
    root.attributes('-topmost', True)
    
    label = tk.Label(root, text="未知人脸，即将锁屏！", font=('Arial', 14), fg='red')
    label.pack(pady=10)
    
    countdown_label = tk.Label(root, text="2", font=('Arial', 36), fg='red')
    countdown_label.pack()
    
    def update_countdown(count):
        if count > 0:
            countdown_label.config(text=str(count))
            root.after(1000, update_countdown, count - 1)
        else:
            root.destroy()
            ctypes.windll.user32.LockWorkStation()
    
    root.after(1000, update_countdown, 1)
    root.mainloop()


def detect_faces_from_camera():
    """
    从摄像头实时检测人脸并识别（每秒检测一次）
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return None
    
    print("实时检测中，每秒检测一次，按q退出")
    last_detect_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break
        
        # 每秒检测一次
        current_time = time.time()
        if current_time - last_detect_time >= 1.0:
            last_detect_time = current_time
            
            # 转换为PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 检测所有人脸
            faces = mtcnn(image)
            
            if faces is not None:
                print(f"\n检测到 {len(faces)} 张人脸")
                
                # 使用模型预测
                with torch.no_grad():
                    outputs = model(faces)
                    val, idx = torch.softmax(outputs, dim=1).max(dim=1)
                    for i, pred in enumerate(idx):
                        prob = val[i].item()
                        pred_class = pred.item()
                        if prob < 0.8:
                            name = "unknown"
                            print("  未知人脸，显示倒计时弹框...")
                            show_countdown_and_lock()
                        else:
                            name = reverse_label_map[pred_class]
                        print(f"  人脸 {i+1}: {name} (置信度: {prob:.2%})")
                    # for i, face in enumerate(faces):
                    #     face_input = face.unsqueeze(0).to(device)
                    #     output = model(face_input)
                    #     pred = output.argmax(dim=1).item()
                    #     prob = torch.softmax(output, dim=1).max().item()
                    #     if prob < 0.8:
                    #         name = "unknown"
                    #         print("  未知人脸，执行锁屏...")
                    #         ctypes.windll.user32.LockWorkStation()
                    #     else:
                    #         name = reverse_label_map[pred]
                    #     print(f"  人脸 {i+1}: {name} (置信度: {prob:.2%})")
        
        # 显示视频流
        cv2.imshow("Camera", frame)
        
        # q键退出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_faces_from_camera()
