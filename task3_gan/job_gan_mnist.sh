#!/bin/bash
#SBATCH --job-name=comp3710-gan-mnist
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=gan_mnist_%j.out
#SBATCH --error=gan_mnist_%j.err

# Needed for the MNIST download - sbatch jobs don't source ~/.bashrc,
# so the proxy vars must be set explicitly here.
export http_proxy=http://proxy.eait.uq.edu.au:8080/
export https_proxy=http://proxy.eait.uq.edu.au:8080/

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

source $HOME/miniconda3/bin/activate
conda activate torch

python -u train_gan_mnist.py "$@"

echo "Job finished $(date)"