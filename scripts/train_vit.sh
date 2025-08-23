#!/bin/bash -l
#
#SBATCH --job-name=train_mmdet_vit         # Job name
#SBATCH --gres=gpu:a100:1 -p a100                     

#SBATCH --time=24:00:00

#SBATCH --output=../work_dirs/train_mmdet_vit_%j.out  # Output log (%j is job ID)
#SBATCH --error=../work_dirs/train_mmdet_vit_%j.err   # Error log (%j is job ID)

#SBATCH --mail-type=begin        # send email when job begins

#SBATCH --mail-type=end          # send email when job ends

#SBATCH --mail-user=debadrita.mukherjee@fau.de

# Load necessary modules
module purge
module load python/3.12-conda       # Load Python 3.8 with Anaconda
module load cuda/11.8.0               # Load CUDA (adjust version as necessary)

# Activate the Conda environment created with Python 3.8
conda activate openmmlab
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Change to the mmdetection directory
cd /home/vault/iwi5/iwi5234h/Dataset/mmdetection

# Run the training command
python tools/train.py configs/mm_grounding_dino/coco/grounding_dino.py