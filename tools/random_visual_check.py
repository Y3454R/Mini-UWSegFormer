import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import random

folder = "data/SUIM/masks_indexed/train"
files = [f for f in os.listdir(folder) if f.endswith(".png")]
if not files:
    print("No PNG files found in folder:", folder)
    exit()

random_file = random.choice(files)
print(f"Showing mask: {random_file}")

mask_path = os.path.join(folder, random_file)
mask = np.array(Image.open(mask_path))

plt.imshow(mask, cmap="nipy_spectral", interpolation="nearest")
plt.title(f"Indexed Mask Visualization: {random_file}")
plt.colorbar()
plt.axis("off")
plt.show()
