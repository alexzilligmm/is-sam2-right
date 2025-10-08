#!/bin/bash
#SBATCH --job-name=eval_duts
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1 
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=50G

cd ..

module load gcc/12.2.0
module load cuda/12.6

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export TORCH_CUDA_ARCH_LIST="8.0"  
export CC=$(which gcc)
export CXX=$(which g++)

source ./.venv/bin/activate

uv sync

nvidia-smi

HYDRA_FULL_ERROR=1 python scripts/amg.py --model_type facebook/sam2.1-hiera-large