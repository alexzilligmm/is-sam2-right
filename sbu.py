import os
from pathlib import Path
from torchvision.datasets import SBU

# 1️⃣ Define paths
root = Path("/datasets/SBU")
images_dir = root / "Images"
masks_dir = root / "Masks"

# Create folders if they don't exist
images_dir.mkdir(parents=True, exist_ok=True)
masks_dir.mkdir(parents=True, exist_ok=True)

# 2️⃣ Initialize the dataset (downloads automatically if not found)
dataset = SBU(root=root, download=True)

# 3️⃣ Move images into Images/
for photo in dataset.photos:
    src = root / "dataset" / photo
    dst = images_dir / photo
    if src.exists():
        src.rename(dst)

# 4️⃣ Move masks into Masks/ (if they exist)
# Note: You must know the naming convention of the mask files
for photo in dataset.photos:
    mask_name = photo.replace(".jpg", "_mask.png")  # adjust as needed
    src_mask = root / "dataset" / mask_name
    if src_mask.exists():
        dst_mask = masks_dir / mask_name
        src_mask.rename(dst_mask)

print("✅ Images and masks organized successfully!")