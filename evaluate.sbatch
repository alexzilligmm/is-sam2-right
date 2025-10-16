#!/bin/bash
#SBATCH --job-name=eval_colondb
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1 
#SBATCH --output=cluster/logs/%j.out
#SBATCH --error=cluster/logs/%j.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=150G

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


python scripts/amg.py --dataset_name ColonDB --model_type large
