# SFT for UI-Ins

## Data Preparation
All data should be smart resized by Qwen-2.5-VL series and we use the center point as the Ground Truth coordinate. The slove method is as following:
```python
from qwen_vl_utils import smart_resize
from PIL import Image
image = PIL.open("image_path")
width,height = image.width, image.height
new_height,new_width = smart_resize(
    height,
    width,
    factor=processor.image_processor.patch_size * processor.image_processor.merge_size #14*2 for Qwen2.5-VL,
    min_pixels=3136,
    max_pixels=4096*2160
)
new_center_x = old_center_x/width*new_width
new_center_y = old_center_y/height*new_height
```

You should prepare data in a sharegpt format as following:
```json
{
    "id": "idx_xxx", 
    "image_size": [
        [
            "width", 
            "height"
        ]
    ],
    "images": ["image_path"],
    "messages": [
        {
            "content":"", 
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
}
```

Add your data dir path into data_info.json as following:
```json
 "UI-Ins": {
    "file_name": "Your data dir",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
}
```


## Training by llama-factory

For training your own model, you can use the following command:
```bash
export NNODES=...
export NODE_RANK=...
export MASTER_ADDR=...
export MASTER_PORT=...
bash scripts/ui_ins_sft.sh
```
