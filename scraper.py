import os
import re
import time
import requests
import html
import threading
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO

BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_max_index(folder: Path) -> int:
    """获取文件夹中最大的图片序号"""
    if not folder.exists():
        return 0
    indices = []
    for f in folder.iterdir():
        if f.is_file():
            try:
                indices.append(int(f.stem))
            except ValueError:
                continue
    return max(indices, default=0)


def search_images(query: str, count: int = 50) -> list[str]:
    """通过 Bing 图片搜索获取图片 URL 列表，支持分页"""
    all_urls = []
    # 分多页获取，每页 first 参数递增
    for first in range(1, count * 2, 35):
        url = f"https://www.bing.com/images/search?q={quote(query)}&form=HDRSC2&first={first}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  搜索页面请求失败 (first={first}): {e}")
            continue

        # 先进行 HTML解码，处理 &quot; 等转义字符
        page_text = html.unescape(resp.text)

        # 匹配 murl（媒体 URL，图片原始链接）
        urls = re.findall(r'"murl"\s*:\s*"(https?://[^"]+)"', page_text)
        # 备用: 匹配未解码的 &quot; 格式
        if not urls:
            urls = re.findall(r'murl&quot;:&quot;(https?://[^&]+)&quot;', resp.text)
        # 备用: Bing 缩略图
        if not urls:
            urls = re.findall(r'(https://tse\d+-mm\.cn\.bing\.net/th/id/[^"?]+)', page_text)

        all_urls.extend(urls)
        time.sleep(0.5)  # 避免请求过快

    # 去重
    seen = set()
    unique_urls = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls[:count]


def download_image(url: str, save_path: Path, retries: int = 3) -> bool:
    """下载单张图片，支持重试"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()

            # 检查是否是图片
            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                return False

            # 验证下载的内容是否为有效图片
            try:
                img = Image.open(BytesIO(resp.content))
                img.load()
            except Exception:
                return False

            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  下载失败 (重试 {retries} 次): {e}")
    return False


# 用于线程安全的打印
_print_lock = threading.Lock()


def _safe_print(msg: str):
    with _print_lock:
        print(msg)


def _download_task(url: str, save_path: Path, index: int, retries: int = 3) -> tuple[bool, str]:
    """线程池任务：下载单张图片，返回 (是否成功, 信息)"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "")
            if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                return False, f"  [{index}] 非图片内容，跳过"

            # 验证下载的内容是否为有效图片
            try:
                img = Image.open(BytesIO(resp.content))
                img.load()  # 尝试加载像素数据，比 verify() 更宽容
            except Exception:
                return False, f"  [{index}] 内容不是有效图片，跳过"

            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True, f"  [{index}] 已保存: {save_path.name}"
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                return False, f"  [{index}] 下载失败 (重试 {retries} 次): {e}"
    return False, f"  [{index}] 下载失败"


def scrape_person(name: str, query: str = None, count: int = 30, workers: int = 8):
    """为指定人物多线程下载图片到 dataset/{name}/ 目录
    
    Args:
        name: 人物文件夹名
        query: 搜索关键词
        count: 目标下载数量
        workers: 并发线程数（默认 8）
    """
    folder = DATASET_DIR / name
    folder.mkdir(parents=True, exist_ok=True)

    start_index = get_max_index(folder) + 1
    search_query = query or name

    print(f"搜索: {search_query} (目标 {count} 张)")
    urls = search_images(search_query, count * 2)  # 多搜一些，因为有些可能下载失败
    print(f"找到 {len(urls)} 个 URL，开始多线程下载 (workers={workers})...")

    # 准备下载任务列表
    tasks = []
    idx = 0
    for url in urls:
        if idx >= count:
            break
        # 根据 URL 后缀确定扩展名
        ext = ".jpg"
        for e in [".png", ".jpeg", ".jpg", ".webp"]:
            if e in url.lower():
                ext = e
                break
        save_path = folder / f"{start_index + idx}{ext}"
        tasks.append((url, save_path, start_index + idx))
        idx += 1

    # 多线程并发下载
    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_task, url, path, index): (url, path)
            for url, path, index in tasks
        }
        for future in as_completed(futures):
            success, msg = future.result()
            _safe_print(msg)
            if success:
                downloaded += 1

    print(f"完成: {name} 新增 {downloaded} 张图片 (从 {start_index} 开始)\n")


if __name__ == "__main__":
    # 在这里添加要爬取的人物
    scrape_person("xingye", query="周星驰 写真 照片", count=20)
    scrape_person("yangmi", query="杨幂 写真 照片", count=30)
    scrape_person("liuyifei", query="刘亦菲 写真 照片", count=20)
    scrape_person("huge", query="胡歌 写真 照片", count=20)
    scrape_person("chenglong", query="成龙 写真 照片", count=20)
    scrape_person("sunhonglei", query="孙红雷 写真 照片", count=20)
    scrape_person("sunli", query="孙俪 写真 照片", count=20)