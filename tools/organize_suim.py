import os
import random
import shutil

random.seed(42)

# Source directories
SRC_BASE = "SUIM_fix"
TRAIN_VAL_IMAGES = os.path.join(SRC_BASE, "train_val", "images")
TRAIN_VAL_MASKS = os.path.join(SRC_BASE, "train_val", "masks")
TEST_IMAGES = os.path.join(SRC_BASE, "test", "images")
TEST_MASKS = os.path.join(SRC_BASE, "test", "masks")

# Destination directories
DST_BASE = "data/SUIM"
DST_IMAGES = os.path.join(DST_BASE, "images")
DST_MASKS_RGB = os.path.join(DST_BASE, "masks_rgb")
DST_SPLITS = os.path.join(DST_BASE, "splits")

TRAIN_RATIO = 0.85

# --- Delete the destination folder if it exists ---
if os.path.exists(DST_BASE):
    shutil.rmtree(DST_BASE)

# --- Recreate directory structure ---
for folder in [
    DST_BASE,
    DST_IMAGES,
    DST_MASKS_RGB,
    DST_SPLITS,
    os.path.join(DST_IMAGES, "train"),
    os.path.join(DST_IMAGES, "val"),
    os.path.join(DST_IMAGES, "test"),
    os.path.join(DST_MASKS_RGB, "train"),
    os.path.join(DST_MASKS_RGB, "val"),
    os.path.join(DST_MASKS_RGB, "test"),
]:
    os.makedirs(folder, exist_ok=True)

# Split images
all_train_val_imgs = [
    f
    for f in os.listdir(TRAIN_VAL_IMAGES)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
random.shuffle(all_train_val_imgs)
split_idx = int(len(all_train_val_imgs) * TRAIN_RATIO)
train_imgs = all_train_val_imgs[:split_idx]
val_imgs = all_train_val_imgs[split_idx:]

print(f"Train samples: {len(train_imgs)}")
print(f"Validation samples: {len(val_imgs)}")


def copy_files(file_list, src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir):
    for filename in file_list:
        shutil.copy2(
            os.path.join(src_img_dir, filename), os.path.join(dst_img_dir, filename)
        )

        mask_name = os.path.splitext(filename)[0] + ".bmp"
        src_mask_path = os.path.join(src_mask_dir, mask_name)
        dst_mask_path = os.path.join(dst_mask_dir, mask_name)
        if os.path.exists(src_mask_path):
            shutil.copy2(src_mask_path, dst_mask_path)
        else:
            print(f"Warning: Mask not found for image {filename}")


# Copy files
copy_files(
    train_imgs,
    TRAIN_VAL_IMAGES,
    TRAIN_VAL_MASKS,
    os.path.join(DST_IMAGES, "train"),
    os.path.join(DST_MASKS_RGB, "train"),
)
copy_files(
    val_imgs,
    TRAIN_VAL_IMAGES,
    TRAIN_VAL_MASKS,
    os.path.join(DST_IMAGES, "val"),
    os.path.join(DST_MASKS_RGB, "val"),
)
test_imgs = [
    f for f in os.listdir(TEST_IMAGES) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
print(f"Test samples: {len(test_imgs)}")
copy_files(
    test_imgs,
    TEST_IMAGES,
    TEST_MASKS,
    os.path.join(DST_IMAGES, "test"),
    os.path.join(DST_MASKS_RGB, "test"),
)


# Write splits
def write_split_file(file_list, filepath):
    with open(filepath, "w") as f:
        for filename in file_list:
            f.write(os.path.splitext(filename)[0] + "\n")


write_split_file(train_imgs, os.path.join(DST_SPLITS, "train.txt"))
write_split_file(val_imgs, os.path.join(DST_SPLITS, "val.txt"))
write_split_file(test_imgs, os.path.join(DST_SPLITS, "test.txt"))

print("Dataset organization complete!")
