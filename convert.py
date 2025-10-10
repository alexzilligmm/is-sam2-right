import os
import pandas as pd
from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Input/output directories
input_dir = "datasets/COME15K/data"
output_dir = "datasets/COME15K_Extracted"

# Output folders
rgb_out_dir = os.path.join(output_dir, "RGB")
depth_out_dir = os.path.join(output_dir, "Depth")
gt_out_dir = os.path.join(output_dir, "GT")

os.makedirs(rgb_out_dir, exist_ok=True)
os.makedirs(depth_out_dir, exist_ok=True)
os.makedirs(gt_out_dir, exist_ok=True)

# Find only test parquet files
parquet_files = [
    os.path.join(input_dir, f)
    for f in os.listdir(input_dir)
    if f.startswith("test-") and f.endswith(".parquet")
]

print(f"🔍 Found {len(parquet_files)} test parquet files")

def decode_image(data):
    """Decode image from bytes, dict, or base64 string."""
    if isinstance(data, dict) and "bytes" in data:
        data = data["bytes"]
    if isinstance(data, str):
        data = base64.b64decode(data)
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError(f"Unsupported image data type: {type(data)}")
    return Image.open(BytesIO(data))

def process_row(args):
    i, row, rgb_col, depth_col, gt_col, rgb_out_dir, depth_out_dir, gt_out_dir, global_index = args
    try:
        # RGB
        rgb_img = decode_image(row[rgb_col])
        rgb_img.save(os.path.join(rgb_out_dir, f"{global_index:06d}.png"))

        # Depth
        if depth_col:
            depth_img = decode_image(row[depth_col])
            depth_img.save(os.path.join(depth_out_dir, f"{global_index:06d}.png"))

        # GT (mask)
        if gt_col:
            gt_img = decode_image(row[gt_col])
            gt_img.save(os.path.join(gt_out_dir, f"{global_index:06d}.png"))

        return True
    except Exception as e:
        print(f"⚠️ Skipping row {i}: {e}")
        return False

# Global index for unique filenames
global_index = 0

for pq_file in parquet_files:
    print(f"\n📂 Processing {pq_file}")
    df = pd.read_parquet(pq_file)
    print(f"Columns: {list(df.columns)}")

    # Identify columns
    rgb_col = next((c for c in df.columns if "rgb" in c.lower() or "image" in c.lower()), None)
    depth_col = next((c for c in df.columns if "depth" in c.lower()), None)
    gt_col = next((c for c in df.columns if c.lower() in ["gt", "mask", "label"]), None)

    if rgb_col is None:
        print("⚠️ No RGB column found, skipping this parquet file.")
        continue

    tasks = [
        (i, row, rgb_col, depth_col, gt_col, rgb_out_dir, depth_out_dir, gt_out_dir, global_index + i)
        for i, row in df.iterrows()
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_row, tasks), total=len(tasks)))

    global_index += len(df)

print(f"\n✅ Extraction complete.")
print(f"📁 RGB images saved to:   {os.path.abspath(rgb_out_dir)}")
print(f"📁 Depth maps saved to:   {os.path.abspath(depth_out_dir)}")
print(f"📁 GT masks saved to:     {os.path.abspath(gt_out_dir)}")