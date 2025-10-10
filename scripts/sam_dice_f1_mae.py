import concurrent.futures
import os
import logging
import argparse
import os
import numpy as np
from skimage import io
from skimage.util import img_as_float
# from skimage.measure import compare_ssim
from tqdm import tqdm
import sys
import time
import shutil
from PIL import Image

from skimage.transform import resize
os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser("Sam infer")
# Required
parser.add_argument("--root_folder", type=str, default='../sam_output/DUTS/vit_h/', help="Location of the models to load in.")
parser.add_argument('--ground_truth_folder', type=str, default='../dataset/DUTS/test_masks', help='note for this run')
parser.add_argument('--job_name', type=str, default='DUTS_vit_h', help='job_name')

args, unparsed = parser.parse_known_args()

# vit_b vit_h vit_l
def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  # current_path = os.getcwd()
  # print(current_path)

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

# args.job_name = 'DDIM_scale:' + str(args.DDIM_scale) + '_' + 'step_size:' + str(args.step_size)
# args.job_name = 'SOD:' + str(args.sampling_timesteps) +'_' + 'beta_sched:' + str(args.beta_sched)
args.job_name = './results/' + time.strftime("%Y%m%d-%H%M%S-")+ str(args.job_name)
create_exp_dir(args.job_name)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.job_name, 'infer_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def calculate_f_measure(prediction, ground_truth, beta=1):
    prediction = prediction.astype(np.bool)
    ground_truth = ground_truth.astype(np.bool)

    tp = np.sum(prediction & ground_truth)
    fp = np.sum(prediction & ~ground_truth)
    fn = np.sum(~prediction & ground_truth)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_measure = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + 1e-8)
    f1_measure = (beta ** 2) * (precision * recall) / (precision + recall + 1e-8)

    return f1_measure


def binary_loader(path):
    assert os.path.exists(path), f"`{path}` does not exist."
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")

def normalize_pil(pre, gt):
    gt = np.asarray(gt)
    pre = np.asarray(pre)
    gt = gt / (gt.max() + 1e-8)
    gt = np.where(gt > 0.5, 1, 0)
    max_pre = pre.max()
    min_pre = pre.min()
    if max_pre == min_pre:
        pre = pre / 255
    else:
        pre = (pre - min_pre) / (max_pre - min_pre)
    return pre, gt

def dice_coefficient(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice

def cal_iou(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    dice = (intersection) / (union + 1e-8)
    return dice

def mean_absolute_error(gt, pred):
    mae = np.mean(np.abs(pred - gt))
    return mae



def find_best_dice(prediction_folder, ground_truth_file, folder):
    ground_truth = binary_loader(ground_truth_file)
    best_dice = 0
    best_f_measure = 0
    best_mae = 1
    best_prediction_file = None

    for prediction_file in os.listdir(prediction_folder):
        if prediction_file == 'metadata.csv':
            continue
        prediction_path = os.path.join(prediction_folder, prediction_file)
        prediction = binary_loader(prediction_path)
        if prediction.size != ground_truth.size:
            prediction = prediction.resize(ground_truth.size, Image.BILINEAR)
        prediction1, ground_truth1 = normalize_pil(pre=prediction, gt=ground_truth)

        try:
            f_dice = dice_coefficient(ground_truth1, prediction1)
            f_measure = calculate_f_measure(ground_truth1, prediction1)
            mae_measure = mean_absolute_error(ground_truth1, prediction1)
        except ValueError:
            print(prediction_file)
            print(prediction_folder)
        # print(f"Folder: {folder}, Folder: {prediction_file}, f_max_measure: {f_max_measure:.4f}")

        if f_dice > best_dice:
            best_dice = f_dice
            best_prediction_file = prediction_file
            best_mae = mae_measure
            best_f_measure = f_measure

        # if f_m_measure > best_f_m_measure:
        #     best_f_m_measure = f_m_measure
    return best_dice, best_f_measure, best_prediction_file, best_mae

def process_folder(folder, root_folder, ground_truth_folder):
    """
    Worker function executed in a separate process.
    Returns (folder, prediction_folder, best_dice, best_f_measure, best_prediction_file, best_mae)
    or returns (folder, prediction_folder, None, None, None, None) on error/missing files.
    """
    prediction_folder = os.path.join(root_folder, folder)
    ground_truth_file = os.path.join(ground_truth_folder, folder + ".png")
    if not os.path.isdir(prediction_folder):
        # Not a folder with predictions: skip
        logging.warning(f"Skipping {prediction_folder}: not a directory")
        return (folder, prediction_folder, None, None, None, None)
    if not os.path.isfile(ground_truth_file):
        logging.warning(f"Missing ground truth for {folder}: {ground_truth_file}")
        return (folder, prediction_folder, None, None, None, None)

    try:
        best_dice, best_f_measure, best_prediction_file, best_mae = find_best_dice(prediction_folder, ground_truth_file, folder)
        return (folder, prediction_folder, best_dice, best_f_measure, best_prediction_file, best_mae)
    except Exception as e:
        logging.exception(f"Error processing {prediction_folder}: {e}")
        return (folder, prediction_folder, None, None, None, None)

def main():
    folders = sorted(os.listdir(args.root_folder))
    n_workers = max(1, os.cpu_count() -1)

    total_dice = 0.0
    total_f_measure = 0.0
    total_mae = 0.0
    folder_count = 0

    results = []

    logging.info(f"Starting multiprocessing evaluation with {n_workers} workers on {len(folders)} items.")

    # Use ProcessPoolExecutor for CPU-bound work (true multicore)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_folder = {executor.submit(process_folder, folder, args.root_folder, args.ground_truth_folder): folder for folder in folders}
        for future in tqdm(concurrent.futures.as_completed(future_to_folder), total=len(future_to_folder), desc="Processing folders"):
            folder = future_to_folder[future]
            try:
                res = future.result()
            except Exception as e:
                logging.exception(f"Worker failed for {folder}: {e}")
                continue

            # res: (folder, prediction_folder, best_dice, best_f_measure, best_prediction_file, best_mae)
            _, prediction_folder, best_dice, best_f_measure, best_prediction_file, best_mae = res

            if best_dice is None:
                # skip missing/error cases
                continue

            folder_count += 1
            total_dice += float(best_dice)
            total_f_measure += float(best_f_measure)
            total_mae += float(best_mae)

            logging.info(f"prediction_folder: {prediction_folder} | best_prediction_file: {best_prediction_file} | best_dice: {best_dice:.4f} | best_f1_measure: {best_f_measure:.4f} | best_mae: {best_mae:.4f}")

    if folder_count == 0:
        logging.warning("No folders processed successfully. Exiting.")
        return

    average_dice = total_dice / folder_count
    average_f1_measure = total_f_measure / folder_count
    average_mae_measure = total_mae / folder_count

    logging.info(f"Average best Dice {average_dice:.4f} Average best F1 {average_f1_measure:.4f} Average best MAE {average_mae_measure:.4f} number: {folder_count}")


if __name__ == "__main__":
    main()