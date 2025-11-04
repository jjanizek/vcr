#!/bin/bash
#SBATCH --job-name=vcr
#SBATCH --partition=roxanad
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# This source file is part of the VCR project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

ml python/3.9.0
ml cuda/11.7.1

# Activate the virtual environment
source /home/groups/roxanad/sonnet/vcr/flam_env/bin/activate

which python
nvcc --version

export PYTHONPATH="/home/groups/roxanad/sonnet/vcr:$PYTHONPATH"
export TMPDIR=/scratch/users/$USER/tmp
export HF_HOME=/scratch/users/$USER/huggingface
export HF_DATASETS_CACHE=/scratch/users/$USER/huggingface/datasets
export TORCH_HOME=/scratch/users/$USER/torch

# Create all directories
mkdir -p $TMPDIR $HF_HOME $HF_DATASETS_CACHE $TORCH_HOME 
which python

# Run the script
python /home/groups/roxanad/sonnet/vcr/src/experiments/bootstrap_resample_for_pvalues.py