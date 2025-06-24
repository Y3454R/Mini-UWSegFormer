import os
from PIL import Image
import numpy as np


def get_unique_colors_from_folder(folder_path):
    unique_colors = set()

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".bmp"):
            file_path = os.path.join(folder_path, filename)
            mask = Image.open(file_path).convert("RGB")
            arr = np.array(mask)
            pixels = arr.reshape(-1, 3)
            # Add each unique color tuple to the set
            for color in np.unique(pixels, axis=0):
                unique_colors.add(tuple(color))

    return sorted(unique_colors)


folder = "data/SUIM/masks_rgb/train"  # Change this path if needed

all_colors = get_unique_colors_from_folder(folder)
print(f"Found {len(all_colors)} unique RGB colors across all masks in '{folder}':")
for c in all_colors:
    print(c)
