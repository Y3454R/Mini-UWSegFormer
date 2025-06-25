import os
from PIL import Image
from torch.utils.data import Dataset


class SUIMDataset(Dataset):
    def __init__(self, images_dir, masks_dir, split_txt, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        with open(split_txt, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, f"{img_id}.png")  # fallback
        mask_path = os.path.join(self.masks_dir, f"{img_id}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # single-channel indexed

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
