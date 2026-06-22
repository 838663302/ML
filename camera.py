import cv2
import os
from datetime import datetime
import ctypes

def lock_screen():
    """调用 Windows API 实现锁屏"""
    ctypes.windll.user32.LockWorkStation()


def capture_camera(save_dir="captured"):
    """
    打开摄像头，显示视频流，按空格拍照，按 q 退出
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 打开摄像头（0 为默认摄像头）
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("摄像头已打开")
    print("按 空格 拍照，按 q 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break
        
        # 显示视频流
        cv2.imshow("Camera", frame)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        # 空格键拍照
        if key == ord(' '):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"photo_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"照片已保存: {filename}")
        
        # q 键退出
        elif key == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头已关闭")


if __name__ == "__main__":
    capture_camera()

