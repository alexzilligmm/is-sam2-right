#!/bin/bash
#SBATCH --job-name=score
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1 
#SBATCH --output=cluster/logs/%j.out
#SBATCH --error=cluster/logs/%j.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=50G

source ./.venv/bin/activate

nvidia-smi

module load gcc/12.2.0
module load cuda/12.6

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export TORCH_CUDA_ARCH_LIST="8.0"  
export HF_HUB_CACHE=/leonardo_scratch/fast/IscrC_USAE/.cache
export CC=$(which gcc)
export CXX=$(which g++)

DATASET_NAME="SBU"

python scripts/sam_dice_f1_mae.py --root_folder is-sam2-right/eff_sam_output/$DATASET_NAME/base_large --ground_truth_folder datasets/$DATASET_NAME/Masks --job_name eff_$DATASET_NAME