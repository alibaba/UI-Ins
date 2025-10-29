from PIL import Image
import imghdr
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 线程安全的计数器
class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

def get_image_info(image_path, counter=None):
    """获取图像的详细信息（线程安全）"""
    info = {
        "path": image_path,
        "size_bytes": os.path.getsize(image_path),
        "last_modified": datetime.fromtimestamp(os.path.getmtime(image_path)).isoformat(),
        "format": None,
        "dimensions": None,
        "mode": None,
        "is_valid": False,
        "error": None
    }
    
    try:
        format_type = imghdr.what(image_path)
        if format_type is None:
            info["error"] = "Not a valid image file"
            return info
            
        with Image.open(image_path) as img:
            img.load()  # 强制加载图像数据
            info.update({
                "format": img.format,
                "dimensions": img.size,
                "mode": img.mode,
                "is_valid": True
            })
    except Exception as e:
        info["error"] = str(e)
    finally:
        if counter:
            counter.increment()
            if counter.value % 100 == 0:  # 每处理100个文件打印进度
                print(f"\rProcessed {counter.value} images...", end="", flush=True)
    
    return info

def validate_images_in_directory(directory, output_file=None, max_workers=8):
    """多线程验证目录中的所有图像文件"""
    results = {
        "valid_images": [],
        "invalid_images": [],
        "summary": {
            "total": 0,
            "valid": 0,
            "invalid": 0
        }
    }
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    image_paths = []
    
    # 收集所有图像文件路径
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("No image files found in the directory.")
        return results
    
    # 初始化线程安全计数器
    counter = Counter()
    total_files = len(image_paths)
    print(f"Found {total_files} image files. Starting validation with {max_workers} threads...")
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_image_info, path, counter): path for path in image_paths}
        
        for future in as_completed(futures):
            try:
                info = future.result()
                if info["is_valid"]:
                    results["valid_images"].append(info)
                else:
                    results["invalid_images"].append(info)
            except Exception as e:
                print(f"\nError processing {futures[future]}: {str(e)}")
    
    # 更新统计信息
    results["summary"]["total"] = len(results["valid_images"]) + len(results["invalid_images"])
    results["summary"]["valid"] = len(results["valid_images"])
    results["summary"]["invalid"] = len(results["invalid_images"])
    
    # 打印报告
    print("\n\nValidation Summary:")
    print(f"Total images checked: {results['summary']['total']}")
    print(f"Valid images: {results['summary']['valid']} ({results['summary']['valid']/results['summary']['total']:.1%})")
    print(f"Invalid images: {results['summary']['invalid']} ({results['summary']['invalid']/results['summary']['total']:.1%})")
    
    if results["invalid_images"]:
        print("\nTop 10 invalid images:")
        for img in results["invalid_images"][:10]:
            print(f"- {img['path']}: {img['error']}")
        if len(results["invalid_images"]) > 10:
            print(f"... and {len(results['invalid_images']) - 10} more")
    
    # 保存报告
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed report saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    # 配置参数
    dataset_name = "seeclick"
    image_directory = f"/home/admin/wangyue/data_utils/aguvis-stage1/data/aguvis/images/{dataset_name}/seeclick_web_imgs"  # 替换为你的路径
    print(image_directory)
    output_file = f"{dataset_name}_image_validation_report.json"
    max_workers = 64  # 根据CPU核心数调整（建议核心数×2）
    
    # 运行验证
    results = validate_images_in_directory(
        directory=image_directory,
        output_file=output_file,
        max_workers=max_workers
    )
