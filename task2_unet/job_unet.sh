#!/bin/bash
#SBATCH --job-name=comp3710-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=unet_%j.out
#SBATCH --error=unet_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate torch

# Pass any extra args through, e.g.:
#   sbatch job_unet.sh --run_name unet_baseline --epochs 40
python -u train_unet.py "$@"

echo "Job finished $(date)"