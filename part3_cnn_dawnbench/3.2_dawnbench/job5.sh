#!/bin/bash
#SBATCH --job-name=comp3710-gpudata
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=part32gpu_%j.out
#SBATCH --error=part32gpu_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"

export http_proxy=http://proxy.eait.uq.edu.au:8080/
export https_proxy=http://proxy.eait.uq.edu.au:8080/

source $HOME/miniconda3/bin/activate
conda activate torch

python -u lab2_part3.2_train_gpu.py
