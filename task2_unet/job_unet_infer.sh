#!/bin/bash
#SBATCH --job-name=comp3710-unet-infer
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=unet_infer_%j.out
#SBATCH --error=unet_infer_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate torch

python -u infer_unet.py "$@"

echo "Job finished $(date)"