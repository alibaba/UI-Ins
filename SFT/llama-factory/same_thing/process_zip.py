import os
import json
from tqdm import tqdm

def find_jsonl_files(root_dir):
    """递归查找所有.jsonl文件"""
    jsonl_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jsonl'):
                print(os.path.join(root, file))
                jsonl_files.append(os.path.join(root, file))
    return jsonl_files

def modify_images_path(data):
    """修改images路径中的Jedi_new为Jedi_LF"""
    if isinstance(data, dict):
        if 'images' in data:
            if isinstance(data['images'], str):
                data['images'] = data['images'].replace('Jedi_new', 'Jedi_LF')
            elif isinstance(data['images'], list):
                data['images'] = [img.replace('Jedi_new', 'Jedi_LF') for img in data['images']]
        # 递归处理嵌套字典
        for key in data:
            if isinstance(data[key], (dict, list)):
                modify_images_path(data[key])
    elif isinstance(data, list):
        # 递归处理列表中的每个元素
        for item in data:
            modify_images_path(item)

def process_jsonl_file(file_path):
    """处理单个jsonl文件"""
    try:
        # 读取数据
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
        
        # 修改数据
        modified = False
        for line in lines:
            original_images = str(line.get('images', ''))
            modify_images_path(line)
            if str(line.get('images', '')) != original_images:
                modified = True
        
        # 如果有修改则保存
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            return True
        return False
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

def main():
    root_dir = '/braincoder-extreme-nas/datasets/Jedi_LF/datasets/aguvis++/seeclick_mi_ui_tars_cleaned'
    jsonl_files = find_jsonl_files(root_dir)
    
    print(f"共找到 {len(jsonl_files)} 个jsonl文件")
    
    modified_count = 0
    for file_path in tqdm(jsonl_files, desc="处理文件中"):
        if process_jsonl_file(file_path):
            modified_count += 1
    
    print(f"处理完成，共修改了 {modified_count}/{len(jsonl_files)} 个文件")

if __name__ == '__main__':
    main()