import os
import shutil
import numpy as np
from PIL import Image

COLOR_CLASS_MAP = {
    (0, 0, 0): 0,  # BW: Background waterbody
    (0, 0, 255): 1,  # HD: Human divers
    (0, 255, 0): 2,  # PF: Plants/sea-grass
    (0, 255, 255): 3,  # WR: Wrecks/ruins
    (255, 0, 0): 4,  # RO: Robots/instruments
    (255, 0, 255): 5,  # RI: Reefs and invertebrates
    (255, 255, 0): 6,  # FV: Fish and vertebrates
    (255, 255, 255): 7,  # SR: Sand/sea-floor & rocks
}


def rgb_to_nearest_class(rgb_pixel, colors, color_to_idx):
    # Compute squared Euclidean distances between pixel and all known colors
    dists = np.sum((colors - rgb_pixel) ** 2, axis=1)
    idx_min = np.argmin(dists)
    return color_to_idx[tuple(colors[idx_min])]


def convert_mask_nearest(input_path, output_path):
    mask = Image.open(input_path).convert("RGB")
    arr = np.array(mask)
    h, w, _ = arr.shape
    result = np.zeros((h, w), dtype=np.uint8)

    # Prepare known colors array for fast distance computation
    known_colors = np.array(list(COLOR_CLASS_MAP.keys()))
    color_to_idx = COLOR_CLASS_MAP

    unmapped_pixels = 0
    total_pixels = h * w

    for i in range(h):
        for j in range(w):
            pixel = arr[i, j]
            # Check if exact match
            if tuple(pixel) in color_to_idx:
                result[i, j] = color_to_idx[tuple(pixel)]
            else:
                # Assign nearest known color index
                result[i, j] = rgb_to_nearest_class(pixel, known_colors, color_to_idx)
                unmapped_pixels += 1

    if unmapped_pixels > 0:
        print(
            f"Warning: {input_path} has {unmapped_pixels} unmapped pixels out of {total_pixels} total."
        )

    Image.fromarray(result).save(output_path)


input_dir = "data/SUIM/masks_rgb/train"
output_dir = "data/SUIM/masks_indexed/train"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".bmp"):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file.replace(".bmp", ".png"))
        convert_mask_nearest(in_path, out_path)
print(f"Converted masks saved to {output_dir}")
