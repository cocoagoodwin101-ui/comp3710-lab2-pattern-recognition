#!/bin/bash
#SBATCH --job-name=comp3710-gan-regen
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=gan_regen_%j.out
#SBATCH --error=gan_regen_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate torch

python -u regenerate_gan_samples.py "$@"

echo "Job finished $(date)"