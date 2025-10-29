import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
def count_image_placeholders(text):
    """Count the number of <image> placeholders in a text."""
    if not isinstance(text, str):
        return 0
    return text.count("<image>")

def check_and_fix_data(input_file):
    """Check and fix data format, ensuring image placeholders match image paths."""
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {input_file}: {e}")
            return False
    
    # Create backup of original file
    backup_file = input_file + '.bak'
    try:
        shutil.copy2(input_file, backup_file)
        print(f"Created backup at: {backup_file}")
    except IOError as e:
        print(f"Error creating backup for {input_file}: {e}")
        return False
    
    # Process the data
    valid_data = []
    invalid_count = 0
    
    for item in tqdm(data):
        if not isinstance(item, dict):
            invalid_count += 1
            continue
            
        # Count image placeholders in all messages
        total_placeholders = 0
        messages = item.get('messages', [])
        images = item.get('images', [])
        
        if not isinstance(messages, list) or not isinstance(images, list):
            invalid_count += 1
            continue
            
        for message in messages:
            if isinstance(message, dict):
                if message.get('role') == 'system':
                    continue
                content = message.get('content')
                total_placeholders += count_image_placeholders(content)
        
        # Check if placeholders match images
        if total_placeholders == len(images) and total_placeholders == 1:
            valid_data.append(item)
        else:
            invalid_count += 1
    
    # Write valid data back to original file
    try:
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(valid_data, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Error writing to {input_file}: {e}")
        return False
    
    # Print summary
    print(f"\nProcessing complete for {input_file}:")
    print(f"Total items: {len(data)}")
    print(f"Valid items: {len(valid_data)}")
    print(f"Invalid items: {invalid_count}")
    
    return True

def find_all_json_files(dir_path):
    """高效查找JSON文件，跳过images文件夹"""
    json_files = []
    for root, dirs, files in os.walk(dir_path):
        # 在遍历前就从dirs中移除images文件夹，避免进入该目录
        if 'images' in dirs:
            dirs.remove('images')  # 修改dirs列表会影响后续遍历
        
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def process_directory(root_dir):
    """Process all JSON files in directory."""
    json_files = find_all_json_files(root_dir)
    # json_files = ['/braincoder-extreme-nas/datasets/Jedi_LF/datasets/aguvis++/android_control-v2/android_control-v2.json']
    print(len(json_files))
    print(json_files)
    
    if not json_files:
        print(f"No JSON files found in directory: {root_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for i, file in enumerate(json_files, 1):
        print(f"{i}. {file}")
    
    # Process all files
    success_count = 0
    for file in json_files:
        print(f"\nProcessing file: {file}")
        if check_and_fix_data(file):
            success_count += 1
    
    print(f"\nFinished processing. Successfully processed {success_count}/{len(json_files)} files.")

if __name__ == "__main__":
    target_dir = "/braincoder-extreme-nas/datasets/Jedi_LF/datasets/aguvis++/android_control-v2"
    
    # Verify directory exists
    if not os.path.isdir(target_dir):
        print(f"Directory not found: {target_dir}")
    else:
        process_directory(target_dir)