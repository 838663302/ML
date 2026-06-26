import cv2
import os
import time
from datetime import datetime
import ctypes
from pathlib import Path

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


def capture_hourly(interval=10):
    """
    每10秒自动拍摄一张照片，保存到 dataset/zhangtao/ 文件夹
    :param interval: 拍摄间隔（秒），默认10秒
    """
    save_dir = Path(__file__).parent / "dataset" / "zhangtao"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取当前文件夹中图片数量作为初始值
    existing = sorted(save_dir.glob("*.jpg"))
    last_num = len(existing)
    print(f"当前文件夹已有 {last_num} 张图片")
    
    print(f"开始每30秒拍照，保存目录: {save_dir}")
    print(f"按 Ctrl+C 停止")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取画面")
                break
            
            # 累加命名
            last_num += 1
            filename = save_dir / f"{last_num}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 照片已保存: {filename} (当前共 {last_num} 张)")
            
            # 等待下一次拍摄
            time.sleep(interval)
                
    except KeyboardInterrupt:
        print("\n已停止拍摄")
    finally:
        cap.release()


if __name__ == "__main__":
    capture_hourly()

