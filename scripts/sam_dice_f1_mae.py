import concurrent.futures
import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
import sys
import time
import shutil
from PIL import Image

# --- Argument Parsing and Setup ---
os.environ['CUDA_VISIBLE_DEVICES']='0' # Note: This line has no effect as the script uses CPU for calculations.
parser = argparse.ArgumentParser("Sam infer")
parser.add_argument("--root_folder", type=str, default='../sam_output/DUTS/vit_h/', help="Location of the prediction folders.")
parser.add_argument('--ground_truth_folder', type=str, default='../dataset/DUTS/test_masks', help='Location of the ground truth masks.')
parser.add_argument('--job_name', type=str, default='DUTS_vit_h', help='A name for this evaluation job.')
args, unparsed = parser.parse_known_args()

# --- Directory and Logging Setup ---
def create_exp_dir(path, scripts_to_save=None):
  """Creates an experiment directory and optionally saves scripts."""
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))
  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

# Create a unique results directory for this run
args.job_name = './results/' + time.strftime("%Y%m%d-%H%M%S-")+ str(args.job_name)
create_exp_dir(args.job_name)

# Configure logging to output to both console and a file
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.job_name, 'infer_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# --- Metric Calculation Functions ---

def calculate_f_measure(prediction, ground_truth, beta=1):
    """Calculates the F-measure."""
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.sum(prediction & ground_truth)
    fp = np.sum(prediction & ~ground_truth)
    fn = np.sum(~prediction & ground_truth)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_measure = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)
    return f_measure

def dice_coefficient(gt, pred):
    """Calculates the Dice coefficient."""
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice

def calculate_iou(gt, pred):
    """Calculates the Intersection over Union (IoU), also known as the Jaccard index."""
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou

def mean_absolute_error(gt, pred):
    """Calculates the Mean Absolute Error."""
    mae = np.mean(np.abs(pred - gt))
    return mae

# --- Image Loading and Processing ---

def binary_loader(path):
    """Loads an image and converts it to grayscale."""
    assert os.path.exists(path), f"`{path}` does not exist."
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")

def normalize_pil(pre, gt):
    """Normalizes prediction and ground truth images to a 0-1 range."""
    gt = np.asarray(gt)
    pre = np.asarray(pre)

    # Binarize ground truth
    gt = gt / (gt.max() + 1e-8)
    gt = np.where(gt > 0.5, 1, 0)

    # Normalize prediction
    max_pre = pre.max()
    min_pre = pre.min()
    if max_pre == min_pre:
        pre = pre / 255 if max_pre == 0 else pre / max_pre
    else:
        pre = (pre - min_pre) / (max_pre - min_pre)
    return pre, gt

# --- Core Evaluation Logic ---

def find_best_dice(prediction_folder, ground_truth_file):
    """
    Finds the prediction file within a folder that maximizes the Dice coefficient.

    Returns the best Dice, and the associated F1, IoU, and MAE scores.
    """
    ground_truth = binary_loader(ground_truth_file)
    best_dice = 0
    best_f_measure = 0
    best_iou = 0
    best_mae = 1
    best_prediction_file = "None"

    for prediction_file in os.listdir(prediction_folder):
        if prediction_file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            prediction_path = os.path.join(prediction_folder, prediction_file)
            prediction = binary_loader(prediction_path)

            if prediction.size != ground_truth.size:
                prediction = prediction.resize(ground_truth.size, Image.BILINEAR)

            prediction_norm, ground_truth_norm = normalize_pil(pre=prediction, gt=ground_truth)

            try:
                f_dice = dice_coefficient(ground_truth_norm, prediction_norm)
                f_measure = calculate_f_measure(ground_truth_norm, prediction_norm)
                iou_measure = calculate_iou(ground_truth_norm, prediction_norm)
                mae_measure = mean_absolute_error(ground_truth_norm, prediction_norm)
            except ValueError:
                logging.error(f"ValueError for {prediction_file} in {prediction_folder}")
                continue

            if f_dice > best_dice:
                best_dice = f_dice
                best_prediction_file = prediction_file
                best_mae = mae_measure
                best_f_measure = f_measure
                best_iou = iou_measure

    return best_dice, best_f_measure, best_iou, best_mae, best_prediction_file

def process_folder(folder, root_folder, ground_truth_folder):
    """
    Worker function for a single folder, executed in a separate process.
    """
    prediction_folder = os.path.join(root_folder, folder)
    ground_truth_file = os.path.join(ground_truth_folder, folder + ".png")

    if not os.path.isdir(prediction_folder):
        logging.warning(f"Skipping {prediction_folder}: not a directory")
        return None
    if not os.path.isfile(ground_truth_file):
        logging.warning(f"Missing ground truth for {folder}: {ground_truth_file}")
        return None

    try:
        best_dice, best_f_measure, best_iou, best_mae, best_prediction_file = find_best_dice(prediction_folder, ground_truth_file)
        return (prediction_folder, best_dice, best_f_measure, best_iou, best_mae, best_prediction_file)
    except Exception as e:
        logging.exception(f"Error processing {prediction_folder}: {e}")
        return None

# --- Main Execution ---

def main():
    folders = sorted(os.listdir(args.root_folder))
    n_workers = max(1, os.cpu_count() - 1)

    # Initialize accumulators
    total_dice = 0.0
    total_f_measure = 0.0
    total_iou = 0.0
    total_mae = 0.0
    folder_count = 0

    logging.info(f"Starting multiprocessing evaluation with {n_workers} workers on {len(folders)} items.")

    # Use a process pool to parallelize the evaluation
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all folders for processing
        future_to_folder = {executor.submit(process_folder, folder, args.root_folder, args.ground_truth_folder): folder for folder in folders}

        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_folder), total=len(future_to_folder), desc="Processing folders"):
            try:
                result = future.result()
            except Exception as e:
                folder = future_to_folder[future]
                logging.exception(f"Worker failed for {folder}: {e}")
                continue

            if result is None:
                continue

            prediction_folder, best_dice, best_f_measure, best_iou, best_mae, best_prediction_file = result

            # Accumulate results
            folder_count += 1
            total_dice += float(best_dice)
            total_f_measure += float(best_f_measure)
            total_iou += float(best_iou)
            total_mae += float(best_mae)

            logging.info(f"Folder: {os.path.basename(prediction_folder)} | Best File: {best_prediction_file} | Dice: {best_dice:.4f} | F1: {best_f_measure:.4f} | IoU: {best_iou:.4f} | MAE: {best_mae:.4f}")

    if folder_count == 0:
        logging.warning("No folders were processed successfully. Exiting.")
        return

    # Calculate and log the final average scores
    average_dice = total_dice / folder_count
    average_f1_measure = total_f_measure / folder_count
    average_iou = total_iou / folder_count
    average_mae_measure = total_mae / folder_count

    logging.info("-" * 50)
    logging.info("Overall Average Scores:")
    logging.info(f"Average Dice: {average_dice:.4f}")
    logging.info(f"Average F1-measure: {average_f1_measure:.4f}")
    logging.info(f"Average IoU: {average_iou:.4f}")
    logging.info(f"Average MAE: {average_mae_measure:.4f}")
    logging.info(f"Total folders processed: {folder_count}")
    logging.info("-" * 50)


if __name__ == "__main__":
    main()