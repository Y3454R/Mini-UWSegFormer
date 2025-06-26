# tools/train.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.uwsegformer import UWSegFormer
from models.ell_loss import edge_loss
from utils.dataset import SUIMDataset
from torchvision.transforms import InterpolationMode

# Device setup
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMAGE_SIZE = (480, 640)
print(f"Using device: {DEVICE}")


# Transform function
def train_transform(image, mask):
    resize_image = transforms.Resize(
        IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR
    )
    resize_mask = transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)

    image = resize_image(image)
    mask = resize_mask(mask)

    image = transforms.ToTensor()(image)
    mask = torch.from_numpy(np.array(mask)).long()
    return image, mask


# Paths
BASE = "data/SUIM"
train_loader = DataLoader(
    SUIMDataset(
        f"{BASE}/images/train",
        f"{BASE}/masks_indexed/train",
        f"{BASE}/splits/train.txt",
        train_transform,
    ),
    batch_size=4,
    shuffle=True,
    num_workers=0,
)
val_loader = DataLoader(
    SUIMDataset(
        f"{BASE}/images/val",
        f"{BASE}/masks_indexed/val",
        f"{BASE}/splits/val.txt",
        train_transform,
    ),
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

# Model, loss, optimizer
model = UWSegFormer(num_classes=8).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Train one epoch
def train_epoch():
    model.train()
    total_loss = 0
    for img, mask in train_loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        optimizer.zero_grad()
        out = model(img)
        ce = criterion(out, mask)
        ell = edge_loss(torch.softmax(out, dim=1), mask)
        loss = ce + 0.1 * ell
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")


# Validate
@torch.no_grad()
def validate():
    model.eval()
    correct, total = 0, 0
    for img, mask in val_loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        pred = model(img).argmax(dim=1)
        correct += (pred == mask).sum().item()
        total += mask.numel()
    acc = correct / total
    print(f"Validation Accuracy: {acc:.4f}")


# Save
def save_model(epoch):
    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"results/checkpoints/epoch_{epoch}.pt")


# Run training
for epoch in range(1, 51):
    print(f"Epoch {epoch}")
    train_epoch()
    validate()
    if epoch % 10 == 0:
        save_model(epoch)
