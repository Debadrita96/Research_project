#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from mmdet.apis import DetInferencer

# Initialize the inferencer
'''inferencer = DetInferencer(
    #model='configs/conditional_detr/conditional-detr_r50_8xb2-50e_coco.py',
    #weights='work_dirs/conditional-detr_r50_8xb2-50e_coco/epoch_150.pth',
    #device='cuda:0')

# Perform inference on an image directory
('data/stenosis/test/images',  # Path to your test images
    out_dir='work_dirs/coco_detection/results/inference/',  # Directory to save results
    no_save_pred=False,  # Save predictions in JSON format
    no_save_vis=False)'''

'''import os
import pickle
import mmcv

# Path to the .pkl file containing predictions
pkl_file_path = "/home/vault/iwi5/iwi5234h/Dataset/mmdetection/work_dirs/coco_detection/grounding_dino/results.pkl"  # Replace with your file path

# Path to save the output
output_directory = "/home/vault/iwi5/iwi5234h/Dataset/mmdetection/work_dirs/coco_detection/grounding_dino"
os.makedirs(output_directory, exist_ok=True)
output_file = os.path.join(output_directory, "highest_scores.txt")

# Step 1: Load predictions from .pkl file
with open(pkl_file_path, "rb") as f:
    predictions = pickle.load(f)

# Labels (adjust according to your dataset)
labels_p = ['1', '2', '3', '4', '5', '6', '7',
    '8', '9', '9a', '10', '10a', '11',
    '12', '12a', '13', '14', '14a',
    '15', '16', '16a', '16b', '16c',
    '12b', '14b', 'stenosis'
]

# Step 2: Function to extract highest scores per label
def get_highest_scores(result):
    """Extract highest confidence scores and their corresponding labels."""
    highest_scores = {}
    pred_instances = result.get('pred_instances', {})
    labels = pred_instances.get('labels', [])
    scores = pred_instances.get('scores', [])

    if len(labels) != len(scores):
        print(f"Warning: Labels and scores length mismatch for result {result}")
        return highest_scores

    for label, score in zip(labels, scores):
        if label >= len(labels_p):
            print(f"Warning: Label {label} is out of range for labels_p with length {len(labels_p)}")
            continue
        label_name = labels_p[label]
        if label_name not in highest_scores or score > highest_scores[label_name]:
            highest_scores[label_name] = score.item()  # Ensure score is a float
    return highest_scores

# Step 3: Process predictions
results = []
for prediction in predictions:
    img_id = prediction.get("img_id")
    if img_id is None:
        print("Warning: Missing img_id in prediction, skipping...")
        continue
    highest_scores = get_highest_scores(prediction)
    results.append((img_id, highest_scores))

# Step 4: Write results to file
with open(output_file, "w") as f:
    for img_id, highest_scores in results:
        f.write(f"Results for Image ID {img_id}:\n")
        for label, score in highest_scores.items():
            f.write(f"  {label}: {score:.4f}\n")
        f.write("\n")

print(f"Results saved to {output_file}")'''

import os
import torch
import torchvision
import mmdet
import mmcv
import mmengine
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.structures import DetDataSample

#config_file = '/home/vault/iwi5/iwi5234h/Dataset/mmdetection/configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py'  # Path to the configuration file
config_file = '/home/vault/iwi5/iwi5234h/Dataset/mmdetection/configs/mm_grounding_dino/coco/grounding_dino.py'  # Path to the configuration file
#checkpoint_file = '/home/vault/iwi5/iwi5234h/Dataset/mmdetection/work_dirs/grounding_dino_swin-t_finetune_16xb4_1x_coco/epoch_12.pth'  # Path to the checkpoint file
checkpoint_file = '/home/vault/iwi5/iwi5234h/Dataset/mmdetection/work_dirs/grounding_dino/epoch_12.pth'  # Path to the checkpoint file

register_all_modules()

# Load model
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# Image folder
image_folder = '/home/vault/iwi5/iwi5234h/Dataset/mmdetection/data/stenosis/test/images'  # Path to the folder containing images
labels_p = ['1', '2', '3', '4', '5', '6', '7',
    '8', '9', '9a', '10', '10a', '11',
    '12', '12a', '13', '14', '14a',
    '15', '16', '16a', '16b', '16c',
    '12b', '14b', 'stenosis'
]  # List of labels corresponding to the dataset classes

# List image files
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]

# Function to get the highest scores per label
def get_highest_scores(result):
    highest_scores = {}
    for label, score in zip(result.pred_instances.labels, result.pred_instances.scores):
        label_name = labels_p[label]
        if label_name not in highest_scores or score > highest_scores[label_name]:
            highest_scores[label_name] = score.item()  # Ensure score is a float
    return highest_scores

# Function to print labels out of range
def get_highest_scores_with_debug(result):
    highest_scores = {}
    for label, score in zip(result.pred_instances.labels, result.pred_instances.scores):
        print(f"Label: {label}, Score: {score}")
        if label >= len(labels_p):
            print(f"Error: Label {label} is out of range for labels_p with length {len(labels_p)}")
            continue
        label_name = labels_p[label]
        if label_name not in highest_scores or score > highest_scores[label_name]:
            highest_scores[label_name] = score.item()
    return highest_scores

# Function to skip labels out of range
def get_highest_scores_skip_out_of_range(result):
    highest_scores = {}
    for label, score in zip(result.pred_instances.labels, result.pred_instances.scores):
        if label >= len(labels_p):
            print(f"Warning: Label {label} is out of range for labels_p with length {len(labels_p)}")
            continue
        label_name = labels_p[label]
        if label_name not in highest_scores or score > highest_scores[label_name]:
            highest_scores[label_name] = score.item()
    return highest_scores

# Results
results = []
for image_path in image_files:
    image = mmcv.imread(image_path, channel_order='rgb')
    result = inference_detector(model, image, text_prompt=labels_p)
    highest_scores = get_highest_scores(result)  # You can switch to other debug functions if needed
    results.append((image_path, highest_scores))

# Output
output_directory = "/home/vault/iwi5/iwi5234h/Dataset/mmdetection/work_dirs/coco_detection/grounding_dino_vit"  # Path to the output directory
os.makedirs(output_directory, exist_ok=True)
output_file = os.path.join(output_directory, 'highest_scores_infer.txt')

# Print and save the results
with open(output_file, 'w') as f:
    for image_path, highest_scores in results:
        print(f"Results for {image_path}:")
        f.write(f"Results for {image_path}:\n")
        for label, score in highest_scores.items():
            print(f"{label}: {score:.4f}")
            f.write(f"{label}: {score:.4f}\n")
        f.write("\n")

print(f"Results saved to {output_file}")