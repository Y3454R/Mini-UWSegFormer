import os
import numpy as np
from PIL import Image

MASK_DIR = "data/SUIM/masks_indexed/train"  # change split if needed
ALLOWED_CLASSES = set(range(8))  # 0 to 7

only_zero_files = []
invalid_pixel_files = []

for file in os.listdir(MASK_DIR):
    if file.endswith(".png"):
        path = os.path.join(MASK_DIR, file)
        mask = np.array(Image.open(path))

        unique_values = set(np.unique(mask))

        # Check for masks that are only background (0)
        if unique_values == {0}:
            only_zero_files.append(file)

        # Check for any values not in 0–7
        if not unique_values.issubset(ALLOWED_CLASSES):
            invalid_pixel_files.append((file, unique_values))

# Print results
print(f"\n🔍 Files with ONLY class 0 (likely all black): {len(only_zero_files)}")
for f in only_zero_files:
    print("  ", f)

print(f"\n❌ Files with INVALID class values: {len(invalid_pixel_files)}")
for f, vals in invalid_pixel_files:
    print(f"  {f} has values {vals}")

print("\n✅ Mask folder check complete.")
