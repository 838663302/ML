import cv2
import torch
import ctypes
import time
import math
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

# 5 个关键点的 3D 模型坐标（毫米）
# 顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
FACE_3D_MODEL = np.array([
    [-30.0, -30.0,  0.0],   # 左眼
    [ 30.0, -30.0,  0.0],   # 右眼
    [  0.0,   0.0, 60.0],   # 鼻尖（突出）
    [-30.0,  60.0,  0.0],   # 左嘴角
    [ 30.0,  60.0,  0.0],   # 右嘴角
], dtype=np.float64)

# 正视镜头的角度阈值（度）
POSE_THRESHOLD = 15

# 畸变系数（简化估算，默认无畸变）
# [k1, k2, p1, p2, k3] 径向畸变 + 切向畸变
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)


def build_camera_matrix(width, height):
    """根据图像宽高构建简化内参矩阵"""
    return np.array([
        [width,           0, width / 2],
        [0,          width, height / 2],
        [0,               0,         1]
    ], dtype=np.float64)


def show_countdown_and_lock(face_count=0):
    """显示10秒倒计时弹框，显示人脸数量。倒计时结束自动锁屏；手动关闭弹窗则取消锁屏。"""
    root = tk.Tk()
    root.title("警告")
    root.geometry("300x150")
    root.attributes('-topmost', True)
    
    should_lock = True  # 是否执行锁屏的标记
    
    label = tk.Label(root, text=f"未知人脸，检测到 {face_count} 张人脸，即将锁屏！",
                     font=('Arial', 12), fg='red')
    label.pack(pady=10)
    
    countdown_label = tk.Label(root, text="10", font=('Arial', 36), fg='red')
    countdown_label.pack()
    
    def do_lock():
        if not should_lock:
            return
        root.destroy()
        ctypes.windll.user32.LockWorkStation()
    
    def update_countdown(count):
        if count > 0:
            countdown_label.config(text=str(count))
            root.after(1000, update_countdown, count - 1)
        else:
            do_lock()
    
    def on_close():
        # 用户手动关闭弹窗：取消锁屏
        nonlocal should_lock
        should_lock = False
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(1000, update_countdown, 9)
    root.mainloop()


def get_head_pose(image_points_2d, camera_matrix):
    """
    用 solvePnP 从 2D 关键点估计头部姿态，返回 pitch/yaw/roll（度）。
    image_points_2d: np.array [5, 2]，MTCNN 5 个关键点的像素坐标
    """
    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, image_points_2d, camera_matrix, DIST_COEFFS,
        flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return None

    # rvec → 旋转矩阵 R
    R, _ = cv2.Rodrigues(rvec)

    # 从 R 提取 pitch/yaw/roll
    pitch = math.degrees(math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2)))
    yaw   = math.degrees(math.atan2(R[1][0], R[0][0]))
    roll  = math.degrees(math.atan2(R[2][1], R[2][2]))

    return pitch, yaw, roll


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
    last_pts = []  # 保存最近一次检测到的关键点，使其在两次检测之间持续显示

    # 从摄像头分辨率估算内参矩阵
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_matrix = build_camera_matrix(width, height)
    
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
            
            # 检测所有人脸，并获取关键点
            # 老版本 API: mtcnn.detect(image, landmarks=True) -> (boxes, probs, points)
            # boxes/points 均为 numpy 数组，points 形状 [N, 5, 2]
            boxes, probs, points = mtcnn.detect(image, landmarks=True)

            if boxes is not None and len(boxes) > 0:
                # 从检测框裁剪出对齐后的人脸张量，用于模型推理
                faces = mtcnn.extract(image, boxes, None)

                # points: [N, 5, 2]，顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
                # 用 solvePnP 计算每张人脸的姿态，过滤掉未正视镜头的
                valid_faces = []
                valid_pts = []
                last_pts = []
                for i in range(len(points)):
                    image_points = np.array(points[i], dtype=np.float64)
                    pose = get_head_pose(image_points, camera_matrix)
                    if pose is None:
                        continue
                    pitch, yaw, roll = pose
                    pts = [(int(x), int(y)) for x, y in points[i]]
                    last_pts.append(pts)

                    is_frontal = (abs(pitch) <= POSE_THRESHOLD and
                                  abs(yaw) <= POSE_THRESHOLD and
                                  abs(roll) <= POSE_THRESHOLD)

                    print(f"  人脸 {i+1}: pitch={pitch:.1f}° yaw={yaw:.1f}° roll={roll:.1f}°"
                          f"  {'正视' if is_frontal else '非正视(跳过)'}")

                    if is_frontal:
                        valid_faces.append(faces[i])
                        valid_pts.append(pts)

                print(f"\n检测到 {len(faces)} 张人脸，正视镜头 {len(valid_faces)} 张")
                
                # 只对正视镜头的人脸做识别
                if len(valid_faces) > 0:
                    valid_faces_tensor = torch.stack(valid_faces)
                    with torch.no_grad():
                        outputs = model(valid_faces_tensor)
                        val, idx = torch.softmax(outputs, dim=1).max(dim=1)
                        for i, pred in enumerate(idx):
                            prob = val[i].item()
                            pred_class = pred.item()
                            if prob < 0.8:
                                name = "unknown"
                                print("  未知人脸，显示倒计时弹框...")
                                show_countdown_and_lock(len(valid_faces))
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
        
        # 在画面上绘制关键点（绿色圆点）
        for pts in last_pts:
            for (x, y) in pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

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
