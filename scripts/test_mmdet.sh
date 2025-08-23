#!/bin/bash -l
#
#SBATCH --job-name=test_mmdet_final         # Job name
#SBATCH --gres=gpu:a100:1 -p a100 

#SBATCH --time=24:00:00

#SBATCH --output=../work_dirs/test_mmdet_final_%j.out  # Output log (%j is job ID)
#SBATCH --error=../work_dirs/test_mmdet_final_imp.err   # Error log (%j is job ID)

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
python tools/test.py configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py work_dirs/conditional-detr_r50_8xb2-50e_coco/epoch_400.pth  --out work_dirs/coco_detection/results.pkl
#python tools/analysis_tools/analyze_results.py configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py work_dirs/coco_detection/conditional_detr/results.pkl results
#python tools/test.py configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py work_dirs/conditional-detr_r50_8xb2-50e_coco/epoch_150.pth --show-dir results/images/
#python tools/analysis_tools/coco_error_analysis.py work_dirs/coco_detection/conditional_detr/test.bbox.json results --ann=data/stenosis/test/annotations/test.json --types='bbox'
#python tools/analysis_tools/confusion_matrix.py configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py work_dirs/coco_detection/conditional_detr/results.pkl results
#python tools/analysis_tools/eval_metric.py configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py work_dirs/coco_detection/conditional_detr/results.pkl




#python tools/test.py configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py work_dirs/grounding_dino_swin-t_finetune_16xb4_1x_coco/epoch_12.pth --out work_dirs/coco_detection/results.pkl
#python tools/analysis_tools/confusion_matrix.py configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py work_dirs/coco_detection/grounding_dino/results.pkl results
#python tools/analysis_tools/analyze_results.py configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py work_dirs/coco_detection/grounding_dino/results.pkl results 
#python tools/test.py configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py work_dirs/grounding_dino_swin-t_finetune_16xb4_1x_coco/epoch_12.pth --show-dir results/images/
#python tools/analysis_tools/coco_error_analysis.py work_dirs/coco_detection/grounding_dino/test.bbox.json results --ann=data/stenosis/test/annotations/test.json --types='bbox'



#python tools/test.py configs/mm_grounding_dino/coco/grounding_dino.py work_dirs/grounding_dino/epoch_12.pth  --out work_dirs/coco_detection/results.pkl
#python tools/analysis_tools/analyze_results.py configs/mm_grounding_dino/coco/grounding_dino.py work_dirs/coco_detection/results.pkl results
#python tools/test.py configs/mm_grounding_dino/coco/grounding_dino.py work_dirs/grounding_dino/epoch_12.pth --show-dir results/images/
#python tools/analysis_tools/coco_error_analysis.py work_dirs/coco_detection/grounding_dino_vit/test.bbox.json results --ann=data/stenosis/test/annotations/test.json --types='bbox'
python tools/analysis_tools/eval_metric.py configs/mm_grounding_dino/coco/grounding_dino.py work_dirs/coco_detection/grounding_dino_vit/results.pkl

