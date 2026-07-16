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
mtcnn = MTCNN(image_size=160, keep_all=True, device=device, thresholds=[0.7, 0.8, 0.8])

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
# 人脸检测置信度阈值（过滤低置信度检测，包括侧脸）
CONFIDENCE_THRESHOLD = 0.9

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


def show_countdown_and_lock(face_count=0, image=None, pts=None):
    """显示10秒倒计时弹框，显示人脸数量、绘制关键点的图像。倒计时结束自动锁屏；手动关闭弹窗则取消锁屏。"""
    from PIL import ImageTk, ImageDraw
    root = tk.Tk()
    root.title("警告")
    root.attributes('-topmost', True)

    should_lock = True  # 是否执行锁屏的标记

    # 文本区域
    label = tk.Label(root, text=f"未知人脸，检测到正视 {face_count} 张人脸，即将锁屏！",
                     font=('Arial', 12), fg='red')
    label.pack(pady=5)

    # 如果有画面，缩放到弹窗内展示，并绘制关键点
    if image is not None:
        # 在拷贝上绘制，避免污染原图
        img_draw = image.copy()
        if pts is not None:
            draw = ImageDraw.Draw(img_draw)
            for face_pts in pts:
                for x, y in face_pts:
                    r = 3
                    draw.ellipse([x-r, y-r, x+r, y+r], fill='red', outline='red')

        # 缩放到合适大小（宽度不超过 400，保持比例）
        w, h = img_draw.size
        max_w = 400
        if w > max_w:
            h = int(h * max_w / w)
            w = max_w
        img_resized = img_draw.resize((w, h))
        tk_img = ImageTk.PhotoImage(img_resized)
        img_label = tk.Label(root, image=tk_img)
        img_label.image = tk_img  # 防止垃圾回收
        img_label.pack(pady=5)

    countdown_label = tk.Label(root, text="10", font=('Arial', 36), fg='red')
    countdown_label.pack(pady=5)
    
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

    # 从旋转矩阵提取 pitch/yaw/roll（ZYX 分解）
    # OpenCV 相机坐标系: X右, Y下, Z朝前
    # pitch = 绕X轴旋转（上下点头）
    # yaw   = 绕Y轴旋转（左右转头）
    # roll  = 绕Z轴旋转（左右歪头）
    sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    if sy < 1e-6:  # gimbal lock 处理
        pitch = math.degrees(math.atan2(-R[1][2], R[1][1]))
        yaw   = math.degrees(math.atan2(-R[2][0], sy))
        roll  = 0.0
    else:
        pitch = math.degrees(math.atan2(R[2][1], R[2][2]))
        yaw   = math.degrees(math.atan2(-R[2][0], sy))
        roll  = math.degrees(math.atan2(R[1][0], R[0][0]))

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
            print("-" * 10, "开始检测", "-" * 10)
            # 检测所有人脸，并获取关键点
            # 老版本 API: mtcnn.detect(image, landmarks=True) -> (boxes, probs, points)
            # boxes/points 均为 numpy 数组，points 形状 [N, 5, 2]
            boxes, probs, points = mtcnn.detect(image, landmarks=True)

            if boxes is not None and len(boxes) > 0:
                if probs is not None:
                    mask = [p >= CONFIDENCE_THRESHOLD for p in probs]
                    boxes = [box for box, m in zip(boxes, mask) if m]
                    probs = [p for p, m in zip(probs, mask) if m]
                    points = [pt for pt, m in zip(points, mask) if m]
                
                # 检查过滤后是否还有检测结果
                if len(boxes) == 0:
                    continue
                
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
                                  abs(yaw) <= POSE_THRESHOLD)

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
                                show_countdown_and_lock(len(valid_faces), image, last_pts)
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
            print("-" * 10, "检测完成", "-" * 10)
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
