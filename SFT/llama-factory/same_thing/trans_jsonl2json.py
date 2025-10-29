import json
import sys
def convert_jsonl_to_json(input_file, output_file):
    """
    将jsonl文件转换为json文件
    
    Args:
        input_file (str): 输入的jsonl文件路径
        output_file (str): 输出的json文件路径
    """
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_jsonl_to_json(input_file, output_file)