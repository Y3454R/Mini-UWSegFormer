# dataset.py
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


class SUIMDataset(Dataset):
    def __init__(self, images_dir, masks_dir, split_txt, transform=None):
        with open(split_txt, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)


def __getitem__(self, idx):
    img_id = self.ids[idx]

    # Try common image extensions
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = os.path.join(self.images_dir, img_id + ext)
        if os.path.exists(img_path):
            break
    else:
        raise FileNotFoundError(
            f"No image found for id {img_id} with extensions jpg/jpeg/png"
        )

    # as mask is PNG
    mask_path = os.path.join(self.masks_dir, img_id + ".png")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"No mask found for id {img_id}")

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)

    if self.transform:
        image, mask = self.transform(image, mask)

    mask = torch.from_numpy(np.array(mask)).long()

    return image, mask

    def __repr__(self):
        return f"SUIMDataset(images_dir={self.images_dir}, masks_dir={self.masks_dir}, num_samples={len(self.ids)})"
