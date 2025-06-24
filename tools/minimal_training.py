import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.uwsegformer import UWSegFormer
from dataset import SUIMDataset  # dataset module
from losses import edge_loss  # ELL loss function

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# Data transforms (example)
def train_transform(image, mask):
    # convert PIL images to tensor, normalize, etc.
    image = transforms.ToTensor()(image)
    # optionally add normalization for underwater images here
    mask = torch.from_numpy(np.array(mask)).long()
    return image, mask


# Dataset and DataLoader
train_dataset = SUIMDataset(
    images_dir="data/SUIM/images/train",
    masks_dir="data/SUIM/masks_indexed/train",
    split_txt="data/SUIM/splits/train.txt",
    transform=train_transform,
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

val_dataset = SUIMDataset(
    images_dir="data/SUIM/images/val",
    masks_dir="data/SUIM/masks_indexed/val",
    split_txt="data/SUIM/splits/val.txt",
    transform=train_transform,
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Initialize model
model = UWSegFormer(num_classes=8)  # 8 SUIM classes
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Mixed precision scaler (optional)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


# Training loop
def train_epoch():
    model.train()
    running_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            ce_loss = criterion(outputs, masks)
            # Assuming outputs is (B, C, H, W), masks is (B, H, W)
            ell = edge_loss(torch.softmax(outputs, dim=1), masks)
            loss = ce_loss + 0.1 * ell

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    print(f"Train Loss: {running_loss / len(train_loader):.4f}")


# Validation loop (basic)
def validate():
    model.eval()
    total_correct = 0
    total_pixels = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()
    print(f"Validation Accuracy: {total_correct / total_pixels:.4f}")


# Main training loop
for epoch in range(1, 51):  # 50 epochs
    print(f"Epoch {epoch}")
    train_epoch()
    validate()
