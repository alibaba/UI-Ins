from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
from PIL import Image
import os
import yaml
import json
import enum
import re
import logging
from tqdm import tqdm
from functools import partial
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import random

# 设置日志配置
def setup_logging():
    # 创建logs目录（如果不存在）
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名，包含时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'process_jedi_{timestamp}.log')
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    logging.info(f"Log file created at: {log_file}")
    return log_file

def merge_consecutive_gpt_messages(data):
    conversations = data["conversations"]
    new_conversations = []
    buffer = []  # 临时存储连续的 gpt 消息
    system = ''
    
    for conv in conversations:
        if conv["from"] == "system":
            system = conv["value"]
        if len(buffer) > 2:
            print(data)
            raise ValueError("buffer length is greater than 2")
        if conv["from"] == "gpt":
            buffer.append(conv["value"])  # 缓存 gpt 消息
        else:
            # 如果 buffer 不为空，说明之前有 gpt 消息，需要合并
            if buffer:
                # 合并所有 gpt 消息，用 \n 连接
                merged_value = "\n".join(buffer)
                # 只保留最后一个 gpt 消息，并更新其 value
                new_conversations.append({"from": "gpt", "value": merged_value})
                buffer = []  # 清空 buffer
            new_conversations.append(conv)  # 添加非 gpt 消息
    
    # 处理末尾可能剩余的 gpt 消息
    if buffer:
        merged_value = "\n".join(buffer)
        new_conversations.append({"from": "gpt", "value": merged_value})
    
    data["conversations"] = new_conversations
    return data, system


def resize_image(image):
# 计算调整后的尺寸
    assert isinstance(image, str) and os.path.exists(image), f"Invalid input image path: {image}"
    image_path = image
    assert os.path.exists(image_path) and os.path.isfile(image_path), f"Invalid input image path: {image_path}"
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, None, None, None, None
    assert isinstance(image, Image.Image), "Invalid input image."
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=28,
        min_pixels=4 * 14 * 14,
        max_pixels=16384 * 14 * 14,)
    # 图片resize
    resized_image = image.resize((resized_width, resized_height))
    return resized_image, image.height, image.width, resized_width, resized_height


def exec_tool_from_gpt(item, factors=[]):
    new_conversations = []
    flag = -1
    for conversation in item['conversations']:
        if conversation['from'] == 'human' and '<image>' in conversation['value']:
            flag += 1
        if conversation['from'] == 'gpt':
            value = conversation['value']
            
            # 1. 尝试提取 JSON 部分
            match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', value, re.DOTALL)
            if not match:
                new_conversations.append(conversation)
                continue
                
            try:
                # 2. 解析 JSON
                data = json.loads(match.group(1))
                
                # 3. 处理坐标
                if 'arguments' in data:
                    # 处理 coordinate
                    if 'coordinate' in data['arguments']:
                        x, y = data['arguments']['coordinate']
                        x_scaled = int(x * factors[flag][0])
                        y_scaled = int(y * factors[flag][1])
                        data['arguments']['coordinate'] = [
                            x_scaled,
                            y_scaled
                        ]
                        # print(f'({x}, {y}) -> ({x_scaled}, {y_scaled}), {factors[flag]}')
                # 4. 保留原始格式（不强制缩进）
                new_json = json.dumps(data, separators=(',', ':'))  # 紧凑格式
                new_value = value.replace(match.group(1), new_json)
                conversation['value'] = new_value
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error processing JSON: {e}")
                return item
                # 出错时保留原始内容
                
        new_conversations.append(conversation)
    
    item['conversations'] = new_conversations
    return item

def exec_box_from_gpt(item, factors=[]):
    new_conversations = []
    flag = -1
    for conversation in item['conversations']:
        if conversation['from'] == 'human' and '<image>' in conversation['value']:
            flag += 1
        if conversation['from'] == 'gpt':
            value = conversation['value']
            
            # 改进后的正则表达式（支持空格/逗号分隔）
            coord_match = re.search(
                r'(?:bounding\s+box:?\s*)?'
                r'[$$$]?\s*'
                r'([-+]?\d*\.\d+|[-+]?\d+)'
                r'(?:\s*[, ]\s*)'
                r'([-+]?\d*\.\d+|[-+]?\d+)'
                r'(?:\s*[, ]\s*)'
                r'([-+]?\d*\.\d+|[-+]?\d+)'
                r'(?:\s*[, ]\s*)'
                r'([-+]?\d*\.\d+|[-+]?\d+)'
                r'[$$$]?',
                value,
                re.VERBOSE
            )
            
            if coord_match:
                try:
                    # 提取并缩放坐标
                    x, y, w, h = map(float, coord_match.groups())
                    scaled_x = int(x * factors[flag][0])
                    scaled_y = int(y * factors[flag][1])
                    scaled_w = int(w * factors[flag][0])
                    scaled_h = int(h * factors[flag][1])
                    
                    # 确定原始格式的分隔符
                    separator = ',' if ',' in coord_match.group(0) else ' '
                    
                    # 重建坐标字符串（保持原始格式）
                    if '(' in coord_match.group(0):
                        new_coord = f"({scaled_x}{separator}{scaled_y}{separator}{scaled_w}{separator}{scaled_h})"
                    elif '[' in coord_match.group(0):
                        new_coord = f"[{scaled_x}{separator}{scaled_y}{separator}{scaled_w}{separator}{scaled_h}]"
                    else:
                        new_coord = f"{scaled_x}{separator}{scaled_y}{separator}{scaled_w}{separator}{scaled_h}"
                    
                    # 保留原始文本结构
                    prefix = value[:coord_match.start()]
                    suffix = value[coord_match.end():]
                    
                    new_conversation = conversation.copy()
                    new_conversation['value'] = prefix + new_coord + suffix
                    new_conversations.append(new_conversation)
                    continue
                
                except (ValueError, IndexError) as e:
                    logging.error(f"Error processing coordinates: {e}")
                    return item
        
        # 非GPT消息或处理失败时保留原样
        new_conversations.append(conversation)
    
    item['conversations'] = new_conversations
    return item

def exec_box_from_human(item, factors=[]):
    new_conversations = []
    flag = -1
    
    for conversation in item['conversations']:
        if conversation['from'] == 'human' and '<image>' in conversation['value']:
            flag += 1
            
        if conversation['from'] == 'human':
            value = conversation['value']
            patterns = [
                # 1. 等号格式 (允许前缀)
                {
                    'regex': r'(\s*(?:bounding box|bbox|box|is)?\s*)(x\s*=\s*(\d+\.?\d*)\s*,\s*y\s*=\s*(\d+\.?\d*)\s*,\s*w\s*=\s*(\d+\.?\d*)\s*,\s*h\s*=\s*(\d+\.?\d*))',
                    'format': 'eq',
                    'prefix': 1,
                    'coords': [3, 4, 5, 6]
                },
                # 2. 标签格式 (x,y,w,h)
                {
                    'regex': r'(\s*(?:bounding box|bbox|box|is)?\s*)(x\s*,\s*y\s*,\s*w\s*,\s*h\s*:\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*))',
                    'format': 'label_xywh',
                    'prefix': 1,
                    'coords': [3, 4, 5, 6]
                },
                # 3. 标签格式 (w,h,x,y)
                {
                    'regex': r'(\s*(?:bounding box|bbox|box|is)?\s*)(w\s*,\s*h\s*,\s*x\s*,\s*y\s*:\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*))',
                    'format': 'label_whxy',
                    'prefix': 1,
                    'coords': [3, 4, 5, 6]
                },
                # 4. 括号格式 (捕获括号类型)
                {
                    'regex': r'(\s*(?:bounding box|bbox|box|is)?\s*)([\[\(]\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*[\]\)])',
                    'format': 'bracket',
                    'prefix': 1,
                    'coords': [3, 4, 5, 6]
                },
                # 5. 无标签四元组 (带前缀)
                {
                    'regex': r'(\s*(?:bounding box|bbox|box|is|:)\s*)((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*))',
                    'format': 'plain',
                    'prefix': 1,
                    'coords': [3, 4, 5, 6]
                },
                # 6. 纯四元组 (无前缀)
                {
                    'regex': r'(\s*)((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*))',
                    'format': 'plain',
                    'prefix': 1,
                    'coords': [3, 4, 5, 6]
                }
            ]
            
            found = False
            for pattern in patterns:
                match = re.search(pattern['regex'], value)
                if match:
                    try:
                        # 提取坐标值
                        prefix = match.group(pattern['prefix'])
                        coord_str = match.group(2)
                        
                        # 提取具体坐标值
                        coords = [float(match.group(i)) for i in pattern['coords']]
                        
                        # 检查缩放因子
                        if flag >= len(factors) or not factors[flag]:
                            scaled_coords = list(map(int, coords))
                        else:
                            w_factor, h_factor = factors[flag]
                            
                            # 根据格式调整缩放顺序
                            if pattern['format'] == 'label_whxy':
                                w, h, x, y = coords
                                scaled_coords = [
                                    int(w * w_factor),
                                    int(h * h_factor),
                                    int(x * w_factor),
                                    int(y * h_factor)
                                ]
                            else:
                                x, y, w, h = coords
                                scaled_coords = [
                                    int(x * w_factor),
                                    int(y * h_factor),
                                    int(w * w_factor),
                                    int(h * h_factor)
                                ]
                        
                        # 生成替换字符串
                        if pattern['format'] == 'eq':
                            replaced_coord = f"x={scaled_coords[0]}, y={scaled_coords[1]}, w={scaled_coords[2]}, h={scaled_coords[3]}"
                        elif pattern['format'] == 'label_xywh':
                            replaced_coord = f"x, y, w, h: {scaled_coords[0]}, {scaled_coords[1]}, {scaled_coords[2]}, {scaled_coords[3]}"
                        elif pattern['format'] == 'label_whxy':
                            replaced_coord = f"w, h, x, y: {scaled_coords[0]}, {scaled_coords[1]}, {scaled_coords[2]}, {scaled_coords[3]}"
                        elif pattern['format'] == 'bracket':
                            # 保留原始括号类型
                            bracket_open = '[' if '[' in coord_str else '('
                            bracket_close = ']' if '[' in coord_str else ')'
                            replaced_coord = f"{bracket_open}{scaled_coords[0]}, {scaled_coords[1]}, {scaled_coords[2]}, {scaled_coords[3]}{bracket_close}"
                        else:  # plain
                            replaced_coord = f"{scaled_coords[0]}, {scaled_coords[1]}, {scaled_coords[2]}, {scaled_coords[3]}"
                        
                        # 组合前缀和替换后的坐标
                        replaced = prefix + replaced_coord
                        
                        # 保留上下文替换
                        new_value = value.replace(match.group(0), replaced, 1)
                        new_conv = conversation.copy()
                        new_conv['value'] = new_value
                        new_conversations.append(new_conv)
                        found = True
                        break
                    
                    except Exception as e:
                        print(f"坐标处理错误: {e}")
                        continue  # 继续尝试其他模式
            
            if not found:
                new_conversations.append(conversation)
        else:
            new_conversations.append(conversation)
    
    item['conversations'] = new_conversations
    return item

def trans_grounding_system(resized_width, resized_height):
    # print(f"resized_width: {resized_width}, resized_height: {resized_height}")  
    system = '''You are a helpful assistant.


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is {{screen_width}}x{{screen_height}}.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button.
* `middle_click`: Click the middle mouse button.
* `double_click`: Double-click the left mouse button.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}, "keys": {"description": "Required only by `action=key`.", "type": "array"}, "text": {"description": "Required only by `action=type`.", "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}, "pixels": {"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}, "args_format": "Format the arguments as a JSON object."}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>'''.replace('{{screen_width}}', str(resized_width)).replace('{{screen_height}}', str(resized_height))
    return system

SYSTYPE = {
    "You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"mobile_use\", \"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 952x2156.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `key`: Perform a key event on the mobile device.\\n    - This supports adb's `keyevent` syntax.\\n    - Examples: \\\"volume_up\\\", \\\"volume_down\\\", \\\"power\\\", \\\"camera\\\", \\\"clear\\\".\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `system_button`: Press the system button.\\n* `open`: Open an app on the device.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"key\", \"click\", \"long_press\", \"swipe\", \"type\", \"system_button\", \"open\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}, \"coordinate2\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=key`, `action=type`, and `action=open`.\", \"type\": \"string\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\", \"type\": \"number\"}, \"button\": {\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\", \"enum\": [\"Back\", \"Home\", \"Menu\", \"Enter\"], \"type\": \"string\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"[:20]: exec_tool_from_gpt,
    "You are an intelligent interface analysis system designed to locate UI elements in screenshots. Your task is to process visual and descriptive information to output accurate bounding box coordinates (x, y, width, height)."[:20]: exec_box_from_gpt,
    "As an expert GUI automation system, your role is to accurately identify and locate interface elements in screenshots. You will receive visual and functional descriptions of UI elements and must determine their exact positions using coordinates (x, y, width, height)."[:20]: exec_box_from_gpt,
    'You are a specialized visual analysis agent focused on GUI automation. When provided with screenshots and element descriptions, your task is to locate and specify the exact position of UI elements using coordinates (x, y, width, height).'[:20]: exec_box_from_gpt,
    'You are an AI assistant trained to locate UI elements in graphical interfaces. Your goal is to analyze screenshots and element descriptions to determine precise bounding box coordinates (x, y, width, height).'[:20]: exec_box_from_gpt,
    'Operating as a precise GUI element locator, your purpose is to analyze screenshots and element descriptions to determine exact positions. You express locations using coordinates (x, y, width, height).'[:20]: exec_box_from_gpt,
    "As a GUI automation assistant, you will receive a screenshot and bounding box coordinates (x, y, w, h). Your task is to generate a detailed, natural description of the UI element within the specified box that enables users to identify it without coordinate references. Please include:\n\n1. Appearance Details: Describe the visual characteristics like shape, color scheme, and design elements\n2. Spatial Context: Explain where the element is positioned in relation to surrounding UI components\n3. Functionality: Describe what the element does and how users can interact with it\n4. Component Classification: Specify the type of UI element (e.g., dropdown menu, checkbox)\n\nUse precise, descriptive language and avoid ambiguous terms. Focus on unique identifying features and contextual relationships that make the element easily recognizable."[:20]: exec_box_from_human,
    """As an interface navigation expert, you will analyze a screenshot containing a UI element defined by a bounding box (x, y, w, h coordinates). Your goal is to provide a comprehensive, natural language description that helps users locate and understand this element without coordinate dependence. Include these key aspects:\n\n1. Visual Features: Elaborate on the element's visual presentation, including its geometric properties, color scheme, and stylistic elements\n2. Spatial Positioning: Detail the element's location by describing its relationship to adjacent UI components\n3. Functional Purpose: Explain the element's role and expected user interactions\n4. Element Classification: Specify the type of UI component (e.g., toggle switch, dropdown list)\n\nEnsure your description is precise and actionable, highlighting unique identifiers and contextual clues that make the element easily distinguishable."""[:20]: exec_box_from_human,
    '''In your role as a GUI interface analyzer, you'll be presented with a screenshot and coordinates (x, y, w, h) defining a bounding box. Your mission is to craft a detailed, user-friendly description of the UI element within this box, enabling easy identification without relying on numerical coordinates. Your analysis should encompass:\n\n1. Visual Characteristics: Thoroughly describe the element's appearance, including its shape, color palette, and design features\n2. Positional Details: Clearly explain the element's location by referencing surrounding UI components and screen layout\n3. User Interaction: Outline the element's intended functionality and how users should interact with it\n4. Component Type: Precisely identify the UI element category (such as slider, radio button, etc.)\n\nProvide concrete, specific descriptions while avoiding vague terminology. Emphasize distinctive visual cues and spatial relationships that make the element easily identifiable.'''[:20]: exec_box_from_human,
    '''Taking on the role of a UI element interpreter, you will receive a screenshot and bounding box coordinates (x, y, w, h). Your assignment is to create a clear, comprehensive description of the UI element contained within the specified area, allowing users to identify it without relying on numerical positions. Your description should include:\n\n1. Appearance Analysis: Detail the element's visual characteristics, including its geometric form, color choices, and styling details\n2. Location Description: Explain the element's position by referencing nearby UI components and layout structure\n3. Interaction Guidelines: Describe the element's purpose and how users should engage with it\n4. Component Identity: Specify the type of UI element (such as navigation bar, form field)\n\nUse precise, descriptive language and avoid vague terms. Emphasize unique visual markers and contextual relationships that make the element easy to spot.'''[:20]: exec_box_from_human,
    """You are a GUI automation agent, and I will give you a screenshot and a bounding box (formatted as x, y, w, h). Your job is to analyze the UI element within that bounding box and provide a clear, natural language description that would help users locate and understand this element without relying on coordinates. Focus on:\n\n1. Visual Description: Describe the element's appearance, including its geometric composition, colors, and styling\n2. Position Information: Explain the element's location relative to other UI components and overall screen placement\n3. Element Function: Detail the element's purpose and expected interactions\n4. Element Type: Identify the specific type of UI component (button, text input, etc.)\n\nBe specific and concrete - avoid vague terms like 'specific area' or 'certain region'. Instead, reference distinctive features, text content, and relationships to other notable elements that would allow users to find this component even without coordinates."""[:20]: exec_box_from_human,
    """Working as a UI navigation guide, you will analyze a screenshot and bounding box coordinates (x, y, w, h). Your objective is to provide a detailed, user-friendly description of the UI element within the specified region, helping users identify it without relying on coordinates. Address these key points:\n\n1. Visual Composition: Detail the element's visual aspects, including its geometry, color scheme, and styling\n2. Spatial Context: Describe the element's position in relation to surrounding interface components\n3. User Interaction: Explain the element's functionality and how users should interact with it\n4. Element Type: Specify the category of UI component (such as progress bar, tooltip)\n\nMaintain precision and clarity in your descriptions, avoiding vague references. Emphasize distinctive characteristics and contextual relationships that make the element easily identifiable."""[:20]: exec_box_from_human,
    """As a GUI component analyst, you will examine a screenshot and bounding box (x, y, w, h values]). Your task is to generate a natural, detailed description of the UI element within the specified region, enabling users to locate it without coordinate references. Your analysis must cover:\n\n1. Visual Details: Describe the element's appearance, including its shape, colors, and design elements\n2. Contextual Position: Explain the element's location relative to surrounding UI components\n3. Functional Role: Detail the element's purpose and expected user interactions\n4. Component Category: Identify the specific type of UI element (e.g., tab panel, scroll bar)\n\nProvide specific, actionable descriptions while avoiding ambiguous terms. Focus on distinctive features and spatial relationships that make the element easily recognizable."""[:20]:exec_box_from_human,
    """You will act as a GUI navigation assistant. Given a screenshot and bounding box (x, y, w, h), provide a comprehensive natural language description of the UI element contained within. Your description should cover:\n\n1. Visual Attributes: Detail the element's visual design, including shapes, colors, and styling\n2. Location Context: Describe the element's placement relative to nearby UI components\n3. Interactive Purpose: Explain the element's role and how users interact with it\n4. UI Element Category: Identify what type of interface component it is\n\nMake your description specific and actionable - avoid vague references and instead focus on distinctive characteristics that would help users locate this element without needing coordinates."""[:20]: exec_box_from_human,
    """Operating as a GUI element descriptor, you will examine a screenshot and a bounding box (x, y, w, h values). Your responsibility is to provide an intuitive, detailed description of the UI component within the specified region, enabling users to locate it without coordinate references. Your description must address:\n\n1. Visual Makeup: Document the element's visual aspects, including its form, coloring, and design patterns\n2. Relative Position: Describe the element's placement in relation to surrounding interface elements\n3. Usage Pattern: Clarify the element's function and how users should interact with it\n4. Interface Element Type: Define the specific category of UI component (e.g., search bar, menu item)\n\nMaintain specificity and clarity, avoiding ambiguous descriptions. Focus on distinctive characteristics and spatial relationships that make the element readily identifiable."""[:20]: exec_box_from_human,


}


jedi_data_path = '/braincoder-extreme-nas/datasets/Jedi/'
# 保存到新的jsonl文件
jedi_name='Jedi_LF_new'
# jedi_jsonl_path_new_base = f'/braincoder-extreme-nas/datasets/{Jedi_new}'
jedi_jsonl_path_new_base = f'/braincoder-extreme-nas/datasets/{jedi_name}'
primus_jedi_base = f'/primus_datasets/datasets/{jedi_name}'



# 读取 YAML 文件
with open(os.path.join(jedi_data_path, 'datasets.yaml'), 'r') as file:
    data = yaml.safe_load(file)

datasets_list = data['datasets']

def read_jsonl(jsonl_path):
    with open(jsonl_path, 'r') as file:
        return [json.loads(line) for line in file]
# datasets_list = [{'images_folder': 'osatlas', 'jsonl_path': 'datasets/aguvis++/osatlas_ui_tars_cleaned/osatlas_ui_tars_cleaned.jsonl', 'sampling_strategy': 'all'}]
double_role_list = []

dir_map = {
            'osatlas_ui_tars_cleaned': '/braincoder-extreme-nas/datasets/OS-Atlas-data/desktop_domain/',
            'seeclick_web_imgs': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/seeclick/seeclick_web_imgs',
            'android_control': '/braincoder-extreme-nas/datasets/aguvis-stage2/images/android_control/android_control/images',
            'guienv': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/guienvs/images',
            'webui350k': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/webui350k/images',
            'omniact': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/omniact/images',
            'seeclick_mi_ui_tars_cleaned': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/seeclick/seeclick_web_imgs',
            'widget_captioning': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/widget_captioning/images',
            'ricosca': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/ricosca/images',
            'ui_refexp': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/ui_refexp/images',
            'ricoig16k': '/braincoder-extreme-nas/datasets/aguvis-stage1/data/aguvis/images/ricoig16k/images',
            'mind2web': '/braincoder-extreme-nas/datasets/aguvis-stage2/images/mind2web/mind2web',
            'guiact-web-single': '/braincoder-extreme-nas/datasets/aguvis-stage2/images/guiact-web-single/guiact-web-single/images',
            'guiact-web-multi': '/braincoder-extreme-nas/datasets/aguvis-stage2/images/guiact-web-multi/guiact-web-multi-v2/images',
            'guide-v2': '/braincoder-extreme-nas/datasets/aguvis-stage2/images/guide/guide-v2/images',
            'coat': '/braincoder-extreme-nas/datasets/aguvis-stage2/images/coat/coat/images', 
        }
dir_map_failed = ['docvqa']
# add=['icons_v0122', 'icons_v0222', 'figma400k','os_layout_data', 'component_v1_130k','doc_images', 'ethercalc_data', 'slide_v1_17k', 'doc_scroll', 'component_library_snap_icon_data','final_1.5m', '']

add_aguvis_img_data = []

def chage_data_sharegpt(item, resized_width, resized_height):
    item['messages'] = []
    img_placeholder_count = 0
    role_map = {
        'human': 'user',
        'gpt': 'assistant',
        'system': 'system'
    }
    for conversation in item['conversations']:
        if conversation['from'] == 'system' and 'You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools>' in conversation['value']:
            conversation['value'] = trans_grounding_system(resized_width, resized_height)
            conversation['from'] = 'system'
        if '<image>' in conversation['value']:
            img_placeholder_count += 1
        item['messages'].append({
                'content': conversation['value'],
                'role': role_map[conversation['from']]
            })
    if 'conversations' in item:
        del item['conversations']
    if 'image' in item:
        del item['image']
    if 'image_id' in item:
        del item['image_id']
    return item, img_placeholder_count


image_size_dict = {}

def process_item(item, tmp_dataset, jedi_data_path, images_folder, new_images_dir, SYSTYPE, images_type):
    try:
        # 存在多个连续的gpt消息，合并到最后一个gpt消息中
        item, system = merge_consecutive_gpt_messages(item)
        # reshape image
        img_path = item['image']
        img_save_path = []
        absolute_paths = []
        img_paths = None
        # 获取前缀
        
        if 'osatlas_ui_tars_cleaned' in tmp_dataset['jsonl_path']:
            prefix = img_path.split('_')[0]
            img_name = '_'.join(img_path.split('_')[1:])
            absolute_paths = [os.path.join('/braincoder-extreme-nas/datasets/OS-Atlas-data/desktop_domain/', prefix, img_name)]
            img_paths = []
        elif images_type in dir_map:
            if isinstance(img_path, str):
                img_save_path = [os.path.join(primus_jedi_base, os.path.dirname(tmp_dataset['jsonl_path']), 'images',img_path)]
                absolute_paths = [os.path.join(dir_map[images_type], img_path)]
                img_paths = [img_path]
            elif isinstance(img_path, list):
                img_save_path = [os.path.join(primus_jedi_base, os.path.dirname(tmp_dataset['jsonl_path']), 'images', img) for img in img_path]
                absolute_paths = [os.path.join(dir_map[images_type], img) for img in img_path]
                img_paths = img_path
            else:
                logging.error(f"img_path is not a string or list: {img_path}")
                raise ValueError(f"img_path is not a string or list: {img_path}")
        else: #jedi自己的数据
            if isinstance(img_path, str):
                # /braincoder-extreme-nas/datasets/Jedi/images/component_v1_130k/{path}
                # /braincoder-extreme-nas/datasets/Jedi/datasets/refusal/refusal_component_v1_130k/refusal_component_v1_130k.jsonl
                # img_save_path:'/primus_datasets/datasets/Jedi_new/datasets/component/generated/component_v1_130k/images/data/toggle-button/other_screenshot/original/ArtisticPalette_1739955969.6706882.png'
                # absolute_paths:/braincoder-extreme-nas/datasets/Jedi/images/component_v1_130k/data/toggle-button/other_screenshot/original/ArtisticPalette_1739955969.6706882.png'
                img_save_path = [os.path.join(primus_jedi_base, os.path.dirname(tmp_dataset['jsonl_path']), 'images', img_path)]
                absolute_paths = [os.path.join(jedi_data_path, 'images', images_folder, img_path)]
                img_paths = [img_path]
            elif isinstance(img_path, list):
                img_save_path = [os.path.join(primus_jedi_base, os.path.dirname(tmp_dataset['jsonl_path']), 'images', p) for p in img_path]
                absolute_paths = [os.path.join(jedi_data_path, 'images', images_folder, p) for p in img_path]
                img_paths = img_path
            else:
                logging.error(f"img_path is not a string or list: {img_path}")
                raise ValueError(f"img_path is not a string or list: {img_path}")

        #确定长宽可以被28*28整除
        factors = []
        if len(absolute_paths) != len(img_paths):
            logging.error(f"len(absolute_paths) != len(img_paths): {len(absolute_paths)} != {len(img_paths)}")
            return None, 1 # missing image
        # assert len(absolute_paths) == len(img_paths)
        image_size = []
        for absolute_path, img_pa in zip(absolute_paths, img_paths) :
            if not os.path.exists(absolute_path):
                logging.warning(f"absolute_path not exists: {absolute_path}")
                return None, 1 # missing image
            resized_image, raw_height, raw_width, resized_width, resized_height = resize_image(absolute_path)
            image_size.append((resized_width, resized_height))
            if resized_image is None:
                logging.warning(f"resized_image is None: {absolute_path}")
                return None, 1 # missing image
            # logging.info(f"raw_size: {raw_width}x{raw_height}, resized_size: {resized_width}x{resized_height}")
            local_save_path = os.path.join(new_images_dir, img_pa)
            local_parent_dir = os.path.dirname(local_save_path)
            if not os.path.exists(local_parent_dir):
                os.makedirs(local_parent_dir, exist_ok=True)
            resized_image.save(local_save_path)
            factors.append([resized_width/raw_width, resized_height/raw_height])

        if system[:20] in SYSTYPE:
            item = SYSTYPE[system[:20]](item, factors=factors)
        item['images'] = img_save_path
        item['image_size'] = image_size

        # 修改item的数据格式，按照sharegpt的格式来组织,并且在system为grounding任务的时候，将resized_width和resized_height添加到item中
        new_item, img_placeholder_count = chage_data_sharegpt(item, resized_width, resized_height)
        # 检查<image>和images数量是否一致
        if img_placeholder_count != len(img_save_path):
            return None, 1 # error
        return new_item, 0 # success
    except Exception as e:
        logging.error(f"Error processing item: {e}", exc_info=True)
        return None, 1 # error
# 统计img_size

def process_jedi(tmp_dataset, jedi_data_path, jedi_jsonl_path_new_base):
    """
    处理jedi数据集
    """
    log_file = setup_logging()
    logging.info(f"Starting to process dataset: {tmp_dataset['jsonl_path']}")
    
    full_jsonl_path = os.path.join(jedi_data_path, tmp_dataset['jsonl_path'])
    logging.info(f"Reading from: {full_jsonl_path}")    
    _dataset = read_jsonl(full_jsonl_path)[:]

    # 随机打乱
    random.shuffle(_dataset)

    logging.info(f"Loaded {len(_dataset)} items from dataset")
    
    new_jsonl_path = os.path.join(jedi_jsonl_path_new_base, tmp_dataset['jsonl_path']).replace('.jsonl', '.json')
    parent_dir = os.path.dirname(new_jsonl_path)
    new_images_dir = os.path.join(parent_dir, "images")

    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(new_images_dir, exist_ok=True)
    logging.info(f"Created output directories: {parent_dir} and {new_images_dir}")
    
    images_folder = tmp_dataset.get('images_folder', 'images')
    logging.info(f"Using images folder: {images_folder}")

    new_dataset = []
    missing_image_count = 0

    process_func = partial(process_item, 
                           tmp_dataset=tmp_dataset, 
                           jedi_data_path=jedi_data_path, 
                           images_folder=images_folder, 
                           new_images_dir=new_images_dir, 
                           SYSTYPE=SYSTYPE,
                           images_type=images_folder)

    # 建议根据任务类型调整进程数
    PROCESSES = min(64, os.cpu_count())  # 不超过 46 个进程
    logging.info(f"Using {PROCESSES} processes for parallel processing")

    with Pool(processes=PROCESSES) as pool:
        with tqdm(total=len(_dataset), desc=f"Processing {os.path.basename(tmp_dataset['jsonl_path'])}") as pbar:
            for result, missing_count_in_item in pool.imap_unordered(process_func, _dataset):
                if result:
                    new_dataset.append(result)
                missing_image_count += missing_count_in_item
                pbar.update(1)

    logging.info(f"Total missing image count for {tmp_dataset['jsonl_path']}: {missing_image_count}")
    logging.info(f"Successfully processed {len(new_dataset)} items")
    logging.info(f"image_size_dict: {image_size_dict}")

    with open(new_jsonl_path, 'w') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)
    logging.info(f"Processed {full_jsonl_path} to {new_jsonl_path}")
    logging.info(f"before: {len(_dataset)}, after: {len(new_dataset)}")
    logging.info(f"Log file saved at: {log_file}")

    return new_jsonl_path


for tmp_dataset in datasets_list:
    images_folder = tmp_dataset.get('images_folder')
    # if 'seeclick_mi_ui_tars_cleaned/seeclick_mi_ui_tars_cleaned.jsonl' not in tmp_dataset['jsonl_path']:
    #     continue
    # print(tmp_dataset.get('images_folder'))
    if 'osatlas' in tmp_dataset['jsonl_path']:
        continue
    # if 'refusal' in tmp_dataset['jsonl_path']:
    #     continue

    # if 'android_control-v2' not in tmp_dataset['jsonl_path']:
    #     continue
    # if images_folder  in dir_map:
    #     continue
    if images_folder in dir_map_failed:
        continue
    process_jedi(tmp_dataset, jedi_data_path, jedi_jsonl_path_new_base)
    logging.info('-'*100)
    logging.info(f"Processed {tmp_dataset['jsonl_path']}")
    logging.info('-'*100)
