#!/bin/bash

# List of datasets
DATASETS=('DUTS' 'COME15K' 'VT5000' 'DIS5K' 'COD10K' 'SBU' 'CDS2K' 'ColonDB')

# Loop over datasets and submit a job for each
for dataset in "${DATASETS[@]}"; do
    sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=base_eval_${dataset}
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=cluster/logs/%j_${dataset}.out
#SBATCH --error=cluster/logs/%j_${dataset}.err
#SBATCH --account=IscrC_SLEY
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=150G

source ./.venv/bin/activate

nvidia-smi

module load gcc/12.2.0
module load cuda/12.6

export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDA_HOME/lib64
export TORCH_CUDA_ARCH_LIST="8.0"
export HF_HUB_CACHE=/leonardo_scratch/fast/IscrC_USAE/.cache
export CC=\$(which gcc)
export CXX=\$(which g++)

python scripts/amg.py --dataset_name $dataset --model_type facebook/sam2.1-hiera-large
EOT
done