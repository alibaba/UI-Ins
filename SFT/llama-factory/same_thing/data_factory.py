# 数据组装
"""
在这里做数据在处理的工作
"""
import json
import random
from tqdm import tqdm

def get_all_data():
    """
    先从所有数据，每个数据获取不超过2W条数据，随机后
    从总的数据集中获取不同尺寸的图片数据，每种size不超过5000条，
    需要打印出每个数据图片size的分布情况
    """

    data_path = '/home/admin/caichenglin/code/llama-factory/data/dataset_info.json'
    data_info = json.load(open(data_path, 'r'))

    all_data = []
    dataset_name_map = {}
    for dataset_name, dataset_info in tqdm(data_info.items()):
        if not dataset_name.endswith('jedi') or 'train' in dataset_name or 'test' in dataset_name or 'refusal' in dataset_name:
            continue
        
        file_name = dataset_info['file_name'].replace('/primus_datasets/datasets/Jedi_LF', '/braincoder-extreme-nas/datasets/Jedi_LF_new')
        data = json.load(open(file_name, 'r'))
        random.shuffle(data)
        data = data[:10000]
        all_data.extend(data)
        dataset_name_map[dataset_name] = len(data)
    print('*'*100)
    print('dataset_name_map: ', dataset_name_map)
    print(len(all_data))
    with open(f'/braincoder-extreme-nas/datasets/Jedi_LF_new/mixture_data/diff_img_size/data_info.json', 'a+') as f:
        json.dump(dataset_name_map, f, indent=4, ensure_ascii=False)
    print('*'*100)
    return all_data

def get_diff_img_size_data():
    """
    从总的数据集中获取不同尺寸的图片数据，每种size不超过5000条
    单挑数据格式为：
    {
        "images": [
            ""
        ],
        "image_size": [
            [
                1092,
                2408
            ]
        ],
        "messages": [
            {
                "content": "",
                "role": "system"
            },
            {
                "content": "",
                "role": "user"
            },
            {
                "content": "",
                "role": "assistant"
            }
        ]
    },
    """

    data = get_all_data()
    random.shuffle(data)
    MAX_SIZE = 5000
    data_map = {}
    result_data = []
    max_token_num = 0
    for item in tqdm(data):
        img_sizes = item['image_size']
        flag = 0
        token_num = 0
        for img_size in img_sizes:
            total_pixels = img_size[0] * img_size[1]
            patch_area = 28 * 28

            if total_pixels % patch_area != 0:
                print(f'{img_size[0]} x {img_size[1]} is not divisible by {patch_area}')
                flag = 1
                break
            tmp_token_num = total_pixels / patch_area
            token_num += tmp_token_num
            if token_num > 12000:
                print(f'{img_size[0]} x {img_size[1]} is too large, token_num: {token_num}')
                flag = 1
                break
            img_size_str = f'{img_size[0]} x {img_size[1]}'
            if img_size_str not in data_map:
                data_map[img_size_str] = 0
            
            if data_map[img_size_str] >= MAX_SIZE:
                flag = 1
                break
            data_map[img_size_str] += 1
            max_token_num = max(max_token_num, token_num)
        if flag == 0:
            result_data.append(item)
    print('*'*100)
    print('data_map: ', data_map)
    with open(f'/braincoder-extreme-nas/datasets/Jedi_LF_new/mixture_data/diff_img_size/data_info.json', 'a+') as f:
        json.dump(data_map, f, indent=4, ensure_ascii=False)
    print('*'*100)
    print('len(result_data): ', len(result_data))
    print('max_token_num: ', max_token_num)
    print('*'*100)
    return result_data

if __name__ == '__main__':
    data = get_diff_img_size_data()
    # 保存数据
    with open(f'/braincoder-extreme-nas/datasets/Jedi_LF_new/mixture_data/diff_img_size/diff_img_size_data_total_{len(data)}.json', 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)