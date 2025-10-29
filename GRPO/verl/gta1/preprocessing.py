import os
import json
import io
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# @dataclass
# class GRPOScriptArguments(ScriptArguments):
#     """
#     Script arguments for the GRPO training script.

#     Args:
#         reward_funcs (`list[str]`):
#             List of reward functions. Possible values: 'accuracy', 'format'.
#     """

#     reward_funcs: list[str] = field(
#         default_factory=lambda: ["accuracy", "format"],
#         metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
#     )
#     max_pixels: Optional[int] = field(
#         default=12845056,
#         metadata={"help": "Maximum number of pixels for the image"},
#     )
#     min_pixels: Optional[int] = field(
#         default=3136,
#         metadata={"help": "Minimum number of pixels for the image"},
#     )
#     image_root: Optional[str] = field(
#         default=None,
#         metadata={"help": "Root directory of the image"},
#     )

# @dataclass
# class GRPOModelConfig(ModelConfig):
#     freeze_vision_modules: bool = False

SYSTEM_PROMPT = '''
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
'''
SYSTEM_PROMPT = SYSTEM_PROMPT.strip()


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, image_root: str, processing_class):
        super(LazySupervisedDataset, self).__init__()
        self.image_root = image_root
        cur_data_dict = []
        if data_path.endswith(".json"):
            with open(data_path, "r") as json_file:
                self.list_data_dict = json.load(json_file)
        else:
            self.list_data_dict = []
            with open(data_path, "r") as f:
                for line in f:
                    self.list_data_dict.append(json.loads(line))
        self.processor=processing_class
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        def make_conversation_image(example, height,width):
            instruction = example['conversations'][0]['value']
            # instruction = instruction.replace("<image>","")
            return {
                # "prompt": [
                #     {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT.format(height=height,width=width)}]},
                #     {
                #         "role": "user",
                #         "content": [
                #             {"type": "image"},
                #             {"type": "text", "text": instruction}
                #         ],
                #     },
                # ],
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(height=height,width=width)},
                    {"role": "user", "content": instruction},
                ],
            }
        example = self.list_data_dict[i]
        image_root = self.image_root   
        image_path = os.path.join(image_root, example['image'])
        image = Image.open(image_path).convert("RGB")
        image_height, image_width =  image.height, image.width
        resized_height, resized_width  = smart_resize(
                    image.height,
                    image.width,
                    factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                    min_pixels=self.processor.image_processor.min_pixels,
                    max_pixels=self.processor.image_processor.max_pixels,
                    )
        image = image.resize((resized_width, resized_height))
        box = example['bbox']
        image = [image]
        

        solution = [box,resized_height/1000,resized_width/1000]

        return {
            'image': image,
            'problem': example['conversations'][0]['value'],
            'solution': solution,
            'prompt': make_conversation_image(example, resized_height,resized_width)['prompt']
        }


def _image_to_bytes(image: Image.Image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


def convert_to_verl_format(dataset: LazySupervisedDataset, output_path: str, head_k: Optional[int] = None, max_workers: int = 4):
    print(f"Using {max_workers} workers to cover {len(dataset)} data samples")
    def process_item(i):
        data_item = dataset[i]
        verl_item = {}
        verl_item["images"] = [{"bytes": _image_to_bytes(data_item["image"][0]), 'path': None}]
        verl_item["data_source"] = "gta1"   
        verl_item["prompt"] = data_item["prompt"]
        verl_item["ability"] = "grounding"
        #  Arrow 要求每一列必须是同质
        ground_truth = data_item["solution"][0] + [data_item["solution"][1], data_item["solution"][2]]
        verl_item["reward_model"] = {"ground_truth": ground_truth, "style": "rule"}
        verl_item["extra_info"] = {'answer': ground_truth, 'index': i, 'question': data_item["problem"], 'split': 'train'}
        return verl_item
    
    if head_k is None:
        total_items = len(dataset)
        indices = range(total_items)
    else:
        total_items = min(head_k, len(dataset))
        indices = range(total_items)
    
    verl_dataset = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(process_item, i): i for i in indices}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(indices), desc="Converting dataset") as pbar:
            for future in as_completed(future_to_index):
                verl_item = future.result()
                verl_dataset.append(verl_item)
                pbar.update(1)
    
    # Sort by index to maintain original order
    verl_dataset.sort(key=lambda x: x["extra_info"]["index"])
    
    verl_df = pd.DataFrame(verl_dataset)
    print(f"Saving to {output_path}, total data samples: {len(verl_dataset)}")
    verl_df.to_parquet(output_path)


if __name__ == "__main__":
    # dataset_name = "/braincoder-extreme-nas/liangyu/GTA1_official/preprocessing/inp.json"
    # image_root = "/braincoder-extreme-nas/liangyu/GTA1_official/preprocessing"
    # max_pixels = 12845056
    # min_pixels = 3136
    # model_path = "/braincoder-extreme-nas/models/Qwen2.5-VL-3B-Instruct"
    # output_path = "/braincoder-extreme-nas/panrong/data/gta1/toy100.parquet"


    dataset_name_prefix = "/braincoder-extreme-nas/panrong/data/gta1/random50k/"
    for dataset_name in ["random50k_train.json", "random50k_val.json"]:
        dataset_name = dataset_name_prefix + dataset_name
        image_root = "/braincoder-extreme-nas/panrong/data/gta1/random50k/images"
        max_pixels = 12845056
        min_pixels = 3136
        model_path = "/braincoder-extreme-nas/models/Qwen2.5-VL-7B-Instruct"
        output_path = dataset_name.replace(".json", ".parquet")

        
        processing_class = AutoProcessor.from_pretrained(model_path,  max_pixels=max_pixels, min_pixels=min_pixels)
        dataset = LazySupervisedDataset(dataset_name, image_root, processing_class)

        convert_to_verl_format(dataset, output_path, max_workers=64)
    
