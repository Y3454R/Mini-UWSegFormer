import os
import numpy as np
from PIL import Image
import shutil

# Define RGB to index mapping for SUIM semantic segmentation classes
COLOR_CLASS_MAP = {
    (0, 0, 0): 0,  # BW - Background / Waterbody (Black)
    (0, 0, 255): 1,  # HD - Human Divers (Blue)
    (0, 255, 0): 2,  # PF - Aquatic Plants / Sea-grass (Green)
    (0, 255, 255): 3,  # WR - Wrecks / Ruins (Sky blue)
    (255, 0, 0): 4,  # RO - Robots / Instruments (Red)
    (255, 0, 255): 5,  # RI - Reefs / Invertebrates (Pink)
    (255, 255, 0): 6,  # FV - Fish and Vertebrates (Yellow)
    (255, 255, 255): 7,  # SR - Sea-floor / Rocks (White)
}


# Function to convert RGB mask to indexed mask
# This function reads an RGB mask image, maps its colors to indices based on COLOR_CLASS_MAP
# and saves the indexed mask as a PNG file.
# Unmapped colors will trigger a warning.
# Input: input_path (str) - path to the input RGB mask image
#        output_path (str) - path to save the indexed mask image
# Output: None (saves the indexed mask image to output_path)
def convert_mask(input_path, output_path):
    mask = Image.open(input_path).convert("RGB")
    arr = np.array(mask)
    h, w, _ = arr.shape
    indexed = np.zeros((h, w), dtype=np.uint8)

    mapped = np.zeros((h, w), dtype=bool)
    for rgb, idx in COLOR_CLASS_MAP.items():
        match = np.all(arr == rgb, axis=-1)
        indexed[match] = idx
        mapped |= match

    if not np.all(mapped):
        print(f"⚠️ Warning: {input_path} contains unmapped colors!")

    Image.fromarray(indexed).save(output_path)


# Input/output directories
split_dirs = ["train", "val", "test"]
for split in split_dirs:
    input_dir = f"data/SUIM/masks_rgb/{split}"
    output_dir = f"data/SUIM/masks_indexed/{split}"

    # Remove and recreate output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".bmp"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace(".bmp", ".png"))
            convert_mask(input_path, output_path)
