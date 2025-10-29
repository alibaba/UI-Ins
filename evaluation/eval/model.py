import torch
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation.configuration_utils import GenerationConfig
import json
import base64
import re
import os
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info, smart_resize
import numpy as np
import pickle as pkl

import multiprocessing as mp

mp.set_start_method('spawn', force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def parse_coordinates(raw_string):
    matches = re.findall(r'\[(\d+),(\d+)\]', raw_string)
    matches = [tuple(map(int, match)) for match in matches]
    if len(matches) == 0:
        return -1,-1
    else:
        return matches[0]
def get_qwen2_5vl_prompt_msg(image, instruction, screen_width, screen_height, magic_prompt=False):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant."
                },
                {
                    "type": "text",
                    "text": """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
Return a json object with a reasoning process in <think></think> tags, a function name and arguments within <tool_call></tool_call> XML tags:
```
<think>
...
</think>
<tool_call>
{"name": "grounding", "arguments": <args-json-object>}
</tool_call>
```
<args-json-object> represents the following item of the action space:
## Action Space{"action": "click", "coordinate": [x, y]}
Your task is to accurately locate a UI element based on the instruction. You should first analyze instruction in <think></think> tags and finally output the function in <tool_call></tool_call> tags.
"""
                }
            ]
        }
    ]
    if magic_prompt:
        messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"Please complete the following tasks by clicking using `click` tool_call: {instruction}"
                }
            ]
        }
        )
    else:
        messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": instruction
                }
            ]
        }
        )
    return messages




class Qwen2_5VLModel():
    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        # Setting default generation config
        self.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True).to_dict()
        self.set_generation_config(
            max_length=2048,
            do_sample=False,
            temperature=0.0
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    def ground_only_positive(self, instruction, image, magic_prompt=False):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
            min_pixels=self.processor.image_processor.min_pixels,
            max_pixels=99999999,
        )
        print("Resized image size: {}x{}".format(resized_width, resized_height))
        resized_image = image.resize((resized_width, resized_height))

        messages = get_qwen2_5vl_prompt_msg(image_path, instruction, resized_width, resized_height, magic_prompt)

        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        guide_text = "<tool_call>\n{\"name\": \"grounding\", \"arguments\": {\"action\": \"click\", \"coordinate\": ["
        text_input = text_input + guide_text
        
        inputs = self.processor(
            text=[text_input],
            images=[resized_image],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        print("Len: ", len(inputs.input_ids[0]))
        generated_ids = self.model.generate(**inputs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]

        response = guide_text + response
        cut_index = response.rfind('}')
        if cut_index != -1:
            response = response[:cut_index + 1]
        print(response)


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        # Parse action and visualize
        try:
            # action = json.loads(response.split('<tool_call>\n')[1].split('\n</tool_call>')[0])
            # coordinates = action['arguments']['coordinate']
            x,y = parse_coordinates(response)
            coordinates = [x,y]
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError("Wrong output format")
            print(point_x, point_y)
            result_dict["point"] = [point_x / resized_width, point_y / resized_height]  # Normalize predicted coordinates
        except (IndexError, KeyError, TypeError, ValueError) as e:
            pass
        
        return result_dict


    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()


class CustomQwen2_5VL_VLLM_Model():
    def __init__(self):
        # Check if the current process is daemonic.
        from multiprocessing import current_process
        process = current_process()
        if process.daemon:
            print("Latest vllm versions spawns children processes, therefore can not be started in a daemon process. Are you using multiprocess.Pool? Try multiprocess.Process instead.")

    def load_model(self, model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct", max_pixels=99999999):  #2007040
        from vllm import LLM
        self.max_pixels = max_pixels
        # self.max_pixels = 12845056
        self.model = LLM(
            model=model_name_or_path,
            limit_mm_per_prompt={"image": 1,},
            trust_remote_code=True,
            dtype="auto",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.90,
            max_model_len=32768,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": self.max_pixels,
            },
        )
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    def set_generation_config(self, **kwargs):
        pass

    def ground_only_positive(self, instruction, image, magic_prompt=False, use_guide_text=True):
        from vllm import SamplingParams
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        # Calculate the real image size sent into the model
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=14 * 2,
            min_pixels=28 * 28,
            max_pixels=self.max_pixels,
        )
        resized_image = image.resize((resized_width, resized_height))

        inputs = [{
            "prompt": get_qwen2_5vl_prompt_msg(
                image_path, 
                instruction, 
                screen_width=resized_width,
                screen_height=resized_height,
                magic_prompt=magic_prompt
            ),
            "multi_modal_data": {"image": resized_image}
        }]
        if use_guide_text:
            guide_text = "<tool_call>\n{\"name\": \"grounding\", \"arguments\": {\"action\": \"click\", \"coordinate\": ["
            inputs[0]["prompt"] += guide_text

        generated = self.model.generate(inputs, sampling_params=SamplingParams(temperature=0.0, max_tokens=256))

        response = generated[0].outputs[0].text.strip()
        print(response)
        if use_guide_text:
            response = """<tool_call>\n{"name": "grounding", "arguments": {"action": "click", "coordinate": [""" + response

        cut_index = response.rfind('}')
        if cut_index != -1:
            response = response[:cut_index + 1]
        print(response)


        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }

        try:
            x,y = parse_coordinates(response)
            coordinates = [x,y]
            if len(coordinates) == 2:
                point_x, point_y = coordinates
            elif len(coordinates) == 4:
                x1, y1, x2, y2 = coordinates
                point_x = (x1 + x2) / 2
                point_y = (y1 + y2) / 2
            else:
                raise ValueError("Wrong output format")
            print(point_x, point_y)
            result_dict["point"] = [point_x / resized_width, point_y / resized_height]  # Normalize predicted coordinates
        except (IndexError, KeyError, TypeError, ValueError) as e:
            pass
        
        return result_dict

    def batch_ground_only_positive(self, instructions, images, magic_prompt=False, use_guide_text=True):
        from vllm import SamplingParams
        assert len(instructions) == len(images), "Number of instructions and images must match"
        
        batch_inputs = []
        resized_images = []
        resized_dimensions = []
        
        # Preprocess all images and prepare inputs
        print("Processing {} images and inputs...".format(len(instructions)))
        for instruction, image in tqdm(zip(instructions, images)):
            if isinstance(image, str):
                image_path = image
                assert os.path.exists(image_path) and os.path.isfile(image_path), f"Invalid input image path: {image_path}"
                image = Image.open(image_path).convert('RGB')
            assert isinstance(image, Image.Image), "Invalid input image."

            # Calculate the real image size sent into the model
            resized_height, resized_width = smart_resize(
                image.height,
                image.width,
                factor=14 * 2,
                min_pixels=28 * 28,
                max_pixels=self.max_pixels,
            )   
            messages = get_qwen2_5vl_prompt_msg(
                image_path, 
                instruction, 
                screen_width=resized_width,
                screen_height=resized_height,
                magic_prompt=magic_prompt
            )

            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if use_guide_text:
                guide_text = "<tool_call>\n{\"name\": \"grounding\", \"arguments\": {\"action\": \"click\", \"coordinate\": ["
                prompt += guide_text

            image_inputs, _ = process_vision_info(messages)
            resized_dimensions.append((resized_width, resized_height))
            
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_inputs} if image_inputs is not None else {},
                "metadata": {"image_path": image_path, "original_data": instruction}
            })

        # print("Batch inputs prepared, running inference...")
        # Run batch inference
        sampling_params = SamplingParams(temperature=0.01, max_tokens=256)
        batch_outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=True)
        
        results = []
        for output, (resized_width, resized_height) in zip(batch_outputs, resized_dimensions):
            response = output.outputs[0].text.strip()
            if use_guide_text:
                response = """<tool_call>\n{"name": "grounding", "arguments": {"action": "click", "coordinate": [""" + response

            result_dict = {
                "result": "positive",
                "format": "x1y1x2y2",
                "raw_response": response,
                "bbox": None,
                "point": None
            }

            cut_index = response.rfind('}')
            if cut_index != -1:
                response = response[:cut_index + 1]

            try:
                x,y = parse_coordinates(response)
                coordinates = [x,y]
                if len(coordinates) == 2:
                    point_x, point_y = coordinates
                elif len(coordinates) == 4:
                    x1, y1, x2, y2 = coordinates
                    point_x = (x1 + x2) / 2
                    point_y = (y1 + y2) / 2
                else:
                    raise ValueError("Wrong output format")
                result_dict["point"] = [point_x / resized_width, point_y / resized_height] # Normalize predicted coordinates
            except (IndexError, KeyError, TypeError, ValueError) as e:
                pass
            
            results.append(result_dict)

        for i in range(len(results)):
            print("index:", i)
            print("prompt:", "\n".join(batch_inputs[i]["prompt"].split("\n")[-4:]))
            print("raw_response:", results[i]["raw_response"])
            print("===="*20)
        
        return results
