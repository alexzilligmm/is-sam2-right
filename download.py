from huggingface_hub import snapshot_download
import os

# Where to save the dataset
target_dir = "datasets/COME15K_test"

# Hugging Face dataset repo
repo_id = "RGBD-SOD/COME15K"

print(f"🔍 Starting download of test split from {repo_id}...")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target_dir,
    allow_patterns=[
        "data/test-*.parquet",  # Only the test Parquet files
    ],
    ignore_patterns=[
        "*.md", "*.txt", "*.zip", "*.pdf", "*.json",
    ],
    tqdm_class=None,  # Set to tqdm if you want a progress bar
)

print(f"✅ Download complete. Files saved under: {os.path.abspath(target_dir)}")

# Optional check
count = 0
for root, _, files in os.walk(target_dir):
    for f in files:
        if f.endswith(".parquet"):
            count += 1
print(f"📦 Found {count} Parquet test files.")