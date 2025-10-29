import json
import os
from PIL import Image
from transformers import AutoTokenizer
import math

# 初始化Qwen 2.5 VL的tokenizer
tokenizer = AutoTokenizer.from_pretrained("/braincoder-extreme-nas/models/Qwen2.5-VL-3B-Instruct/")

def calculate_image_tokens(image_path):
    """计算单张图片的token数"""
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            return math.ceil((w * h) / (28 * 28))
    except Exception as e:
        print(f"Error calculating image tokens: {e}")
        return 0

def process_data_item(data_item, max_token=2048):
    """处理单条数据"""
    # 计算图像token
    image_tokens = 0
    image_paths = data_item.get("images", [])
    for img_path in image_paths:
        if os.path.exists(img_path):
            image_tokens += calculate_image_tokens(img_path)
    
    # 准备消息内容
    messages = data_item.get("messages", [])
    
    # 使用chat template计算文本token
    try:
        # 我们需要先构建一个不包含图像标记的对话
        text_only_messages = []
        for msg in messages:
            # 移除可能存在的<image>标记
            content = msg["content"].replace("<image>", "").strip()
            if content:  # 只保留有内容的message
                text_only_messages.append({"role": msg["role"], "content": content})
        
        if not text_only_messages:
            text_tokens = 0
        else:
            # 使用apply_chat_template计算token数
            chat_template = tokenizer.apply_chat_template(
                text_only_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            text_tokens = chat_template.shape[1]
    except Exception as e:
        print(f"Error calculating text tokens: {e}")
        text_tokens = 0
    
    total_tokens = image_tokens + text_tokens
    
    # 检查是否超过最大token数
    if total_tokens > max_token:
        return None  # 舍弃这条数据
    else:
        return {
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "total_tokens": total_tokens,
            "data": data_item  # 保留原始数据
        }

def process_dataset(data, max_token=2048):
    """处理整个数据集"""
    processed_data = []
    for item in data:
        result = process_data_item(item, max_token)
        if result is not None:
            processed_data.append(result)
    return processed_data

# 示例使用
if __name__ == "__main__":
    # 示例数据
    example_data = [
        {
            "images": [
                "/braincoder-extreme-nas/datasets/Jedi_LF/datasets/aguvis++/android_control-v2/images/0/screenshot_2.png"
            ],
            "messages": [
                {
                    "content": "",
                    "role": "system"
                },
                {
                    "content": "<image>\nWhat is shown in this image?",
                    "role": "user"
                },
                {
                    "content": "This is a screenshot of a mobile app interface.",
                    "role": "assistant"
                }
            ]
        }
    ]
    
    processed = process_dataset(example_data)
    print(json.dumps(processed, indent=2))