#!/bin/bash
#SBATCH --job-name=yolov9_train
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=yolov9_train_%j.out
#SBATCH --error=yolov9_train_%j.err

# Environment setup
source /scratch/apps/packages/anaconda3/etc/profile.d/conda.sh
conda activate yolov9_env

# CUDA path
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

# Go to project directory
cd ~/mtp_project/AeroGuard

# Run training
python train_hpc.py
