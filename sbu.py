import os
from pathlib import Path
from torchvision.datasets import SBU

# Use the current folder as root
root = Path(".") / "datasets" / "SBU"
images_dir = root / "Images"
masks_dir = root / "Masks"

# Create the folders
images_dir.mkdir(parents=True, exist_ok=True)
masks_dir.mkdir(parents=True, exist_ok=True)

# Initialize the dataset (downloads automatically if needed)
dataset = SBU(root=root, download=True)

# Move images into Images/
for photo in dataset.photos:
    src = root / "dataset" / photo
    dst = images_dir / photo
    if src.exists():
        src.rename(dst)

# Move masks into Masks/ (if they exist)
for photo in dataset.photos:
    mask_name = photo.replace(".jpg", "_mask.png")  # adjust if your masks have different names
    src_mask = root / "dataset" / mask_name
    if src_mask.exists():
        dst_mask = masks_dir / mask_name
        src_mask.rename(dst_mask)

print("✅ Images and masks organized successfully!")