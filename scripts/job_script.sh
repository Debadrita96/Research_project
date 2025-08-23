#!/bin/bash
#SBATCH --time=24:00:00                 # Set a time limit of 24 hours
#SBATCH --output=/home/hpc/iwi5/iwi5234h/FAU_Project/global_logs/log_train.out  # Output file for logs
#SBATCH --error=/home/hpc/iwi5/iwi5234h/FAU_Project/global_logs/job_error.log
#SBATCH --mail-type=BEGIN               # Send email when job begins
#SBATCH --mail-type=END                 # Send email when job ends
#SBATCH --mail-user=debadrita.mukherjee@fau.de 


python convert_to_coco.py
