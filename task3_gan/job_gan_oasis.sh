#!/bin/bash
#SBATCH --job-name=comp3710-gan-oasis
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=gan_oasis_%j.out
#SBATCH --error=gan_oasis_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate torch

# First submission:  sbatch job_gan_oasis.sh --run_name gan_oasis --epochs 100
# To continue after hitting the time limit:
#                    sbatch job_gan_oasis.sh --run_name gan_oasis --epochs 100 --resume
python -u train_gan_oasis.py "$@"

echo "Job finished $(date)"