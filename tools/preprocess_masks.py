# Convert RGB BMP masks to class-index PNG masks

import os
import numpy as np
from PIL import Image

COLOR_CLASS_MAP = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 0): 4,
    (255, 0, 255): 5,
    (255, 255, 0): 6,
    (255, 255, 255): 7
}

def convert_mask(input_path, output_path):
    mask = Image.open(input_path).convert("RGB")
    arr = np.array(mask)
    h, w, _ = arr.shape
    result = np.zeros((h, w), dtype=np.uint8)

    for rgb, idx in COLOR_CLASS_MAP.items():
        result[np.all(arr == rgb, axis=-1)] = idx

    Image.fromarray(result).save(output_path)

input_dir = "data/SUIM/masks_rgb/train"
output_dir = "data/SUIM/masks_indexed/train"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".bmp"):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file.replace(".bmp", ".png"))
        convert_mask(in_path, out_path)

