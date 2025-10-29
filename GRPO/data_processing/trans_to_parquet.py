import os
import json
from functools import partial
from PIL import Image
from qwen_vl_utils import smart_resize # Assuming this is a custom utility
from transformers import AutoProcessor
from datasets import load_dataset, DatasetDict
import traceback # For better error logging in multiprocessing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_JSON = "Replace your jsonl here"
FILENAME_PREFIX = "UI_Ins_RL"
TEST_RATIO = 0
MODEL_PATH = "Replace your model path here"

# common configs
IMAGE_ROOT = "Replace your image root here"   # 图片根目录
MAX_PIXELS = 4096 * 2160
MIN_PIXELS = 3136

OUT_DIR = "Replace your result dir here"

SEED = 42
N_PROC = 32 # Number of processes

SYSTEM_PROMPT='''
Replace your training sys prompt here
'''.strip()
# --------------------------------------------------
# 1) Load and split data
# --------------------------------------------------
logging.info(f"Loading dataset from {DATA_JSON}")
try:
    raw_ds = load_dataset("json", data_files=DATA_JSON, num_proc=N_PROC) # Use num_proc for initial loading if beneficial
    logging.info(f"First example: {raw_ds['train'][0]}")

    # If TEST_RATIO=0, all data goes to train
    if TEST_RATIO == 0:
        ds = DatasetDict({"train": raw_ds["train"], "test": raw_ds["train"].select([])})
        logging.info(f"Dataset split: All data in 'train'. Total train samples: {len(ds['train'])}")
    else:
        split_ds = raw_ds["train"].train_test_split(test_size=TEST_RATIO, seed=SEED)
        ds = DatasetDict({"train": split_ds["train"], "test": split_ds["test"]})
        logging.info(f"Dataset split: Train samples: {len(ds['train'])}, Test samples: {len(ds['test'])}")

except Exception as e:
    logging.error(f"Error loading or splitting dataset: {e}")
    exit(1)

# --------------------------------------------------
# 2) Processing function: returns verl_item structure
# --------------------------------------------------
def prepare_verl_item_safe(example, idx, *, split_name, image_root, processor):
    """
    Wrapper for prepare_verl_item to add robust error handling and logging.
    """
    try:
        # ---------- Read Image & Resize ----------
        img_path = os.path.join(image_root, example["image"])
        # Use a context manager for opening the image to ensure it's closed
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            new_h, new_w = smart_resize(
                img.height,
                img.width,
                factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
                min_pixels=processor.image_processor.min_pixels,
                max_pixels=processor.image_processor.max_pixels,
            )
            ori_height = img.height
            ori_width = img.width
            # Resize in-place or assign to ensure memory is managed.
            # Convert to Bytes for better memory management across processes if necessary.
            # However, datasets.map usually handles PIL Image objects well.
            if img.height != new_h or img.width != new_w:
                img = img.resize((new_w, new_h), Image.LANCZOS) # Use LANCZOS for high quality downsampling
            # ---------- Prompt ----------
            instruction = example["conversations"][0]["value"]
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content":f"User's instruction is: "+ instruction},
            ]

            # ---------- Solution & Ground-truth ----------
            box = example["bbox"]                          # [x1, y1, x2, y2]
            solution = [box, new_h/1000 , new_w /1000]   # Keep consistent with original example
            ground_truth = solution[0] + [solution[1], solution[2]]

            # ---------- verl_item ----------
            verl_item = {
                "images": [img],
                "data_source": "gta1",
                "prompt": prompt,
                "ability": "grounding",
                "reward_model": {
                    "ground_truth": ground_truth,
                    "style": "rule",
                },
                "extra_info": {
                    "path": img_path,
                },
            }
            return verl_item
    except Exception as e:
        # Log the error with more context
        logging.error(f"Error processing item {idx} (image: {example.get('image', 'N/A')}): {e}")
        logging.error(traceback.format_exc()) # Print full traceback
        return None # Returning None will effectively filter out this item if `filter` is applied later or if `remove_columns` is intelligent.


# Initialize processor once
logging.info(f"Initializing processor from {MODEL_PATH}")
try:
    processing_class = AutoProcessor.from_pretrained(
        MODEL_PATH,
        max_pixels=MAX_PIXELS,
        min_pixels=MIN_PIXELS,
        use_fast=True,
        trust_remote_code=True # Added for safety with custom models
    )
except Exception as e:
    logging.error(f"Error initializing AutoProcessor: {e}")
    exit(1)

transform_train = partial(
    prepare_verl_item_safe,
    split_name="train",
    image_root=IMAGE_ROOT,
    processor=processing_class,
)
transform_test = partial(
    prepare_verl_item_safe,
    split_name="test",
    image_root=IMAGE_ROOT,
    processor=processing_class,
)

# --------------------------------------------------
# 4) Apply mapping
# --------------------------------------------------
logging.info("Starting processing for 'train' split...")
try:
    ds["train"] = ds["train"].map(
        transform_train,
        with_indices=True,
        remove_columns=ds["train"].column_names,
        num_proc=N_PROC,
        desc="Processing train split", # Add a description for tqdm progress bar
    ).filter(lambda x: x is not None) # Filter out None values from failed items
    logging.info(f"Finished processing 'train' split. New length: {len(ds['train'])}")

except Exception as e:
    logging.error(f"Error during 'train' split mapping: {e}")
    exit(1)


# Only process test dataset if TEST_RATIO > 0
if TEST_RATIO > 0:
    logging.info("Starting processing for 'test' split...")
    try:
        ds["test"] = ds["test"].map(
            transform_test,
            with_indices=True,
            remove_columns=ds["test"].column_names,
            num_proc=N_PROC,
            desc="Processing test split",
        ).filter(lambda x: x is not None)
        logging.info(f"Finished processing 'test' split. New length: {len(ds['test'])}")
    except Exception as e:
        logging.error(f"Error during 'test' split mapping: {e}")
        exit(1)


# --------------------------------------------------
# 5) Store as parquet
# --------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
model_name = MODEL_PATH.strip("/").split("/")[-1] # Robust handling of trailing slash

train_output_path = os.path.join(OUT_DIR, f"{FILENAME_PREFIX}_train_mp{MAX_PIXELS}_{model_name}.parquet")
logging.info(f"Saving 'train' split to {train_output_path}")
try:
    ds["train"].to_parquet(train_output_path)
    logging.info("'train' split saved successfully.")
except Exception as e:
    logging.error(f"Error saving 'train' split: {e}")

if TEST_RATIO > 0:
    val_output_path = os.path.join(OUT_DIR, f"{FILENAME_PREFIX}_val_mp{MAX_PIXELS}_{model_name}.parquet")
    logging.info(f"Saving 'test' split to {val_output_path}")
    try:
        ds["test"].to_parquet(val_output_path)
        logging.info("'test' split saved successfully.")
    except Exception as e:
        logging.error(f"Error saving 'test' split: {e}")

logging.info("Data processing complete.")