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
        print(f"⚠️  Warning: {input_path} contains unmapped colors!")
        return False

    Image.fromarray(indexed).save(output_path)
    return True


split_dirs = ["train", "val", "test"]
for split in split_dirs:
    input_dir = f"data/SUIM/masks_rgb/{split}"
    output_dir = f"data/SUIM/masks_indexed/{split}"

    # Remove and recreate output directory
    if os.path.exists(output_dir):
        print(f"🧹 Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    print(f"📁 Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🔄 Processing split: {split}")
    files = [f for f in os.listdir(input_dir) if f.endswith(".bmp")]
    total = len(files)
    unmapped_count = 0

    for i, file in enumerate(files, start=1):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".bmp", ".png"))
        success = convert_mask(input_path, output_path)
        if not success:
            unmapped_count += 1
        print(f"[{i}/{total}] Processed: {file}", end="\r")

    print(f"\n✅ Finished processing {total} files in '{split}'")
    if unmapped_count:
        print(f"⚠️  {unmapped_count} files had unmapped colors.")
    else:
        print("✅ All masks converted successfully.\n")
