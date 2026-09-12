#!/bin/bash
#SBATCH --job-name=comp3710-vae
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

source $HOME/miniconda3/bin/activate
conda activate torch

# Pass any extra args through, e.g.:
#   sbatch job_vae.sh --run_name beta_low --beta 0.5
python -u train_vae.py "$@"

echo "Job finished $(date)"