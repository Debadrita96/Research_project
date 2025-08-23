#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mmdet.core.evaluation import eval_map
import mmcv
import os

# Define paths
predictions_path = 'work_dirs/coco_detection/results.pkl'
annotations_path = 'data/stenosis/test/annotations/test.json'
output_dir = 'work_dirs/coco_detection/eval_results'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load predictions
predictions = mmcv.load(predictions_path)

# Load ground truth annotations
annotations = mmcv.load(annotations_path)

# Compute mAP
mAP, detailed_results = eval_map(predictions, annotations, iou_thr=0.5, dataset=None)

# Print mAP to console
print(f"mAP: {mAP}")

# Save results to a file
results_file = os.path.join(output_dir, 'eval_results.txt')
with open(results_file, 'w') as f:
    f.write(f"mAP (IoU=0.5): {mAP:.4f}\n\n")
    f.write("Detailed Results:\n")
    for cls_idx, cls_result in enumerate(detailed_results):
        if cls_result is not None:  # Some classes may have no predictions
            f.write(f"Class {cls_idx}: {cls_result['ap']:.4f}\n")
        else:
            f.write(f"Class {cls_idx}: No predictions\n")

print(f"Evaluation results saved to {results_file}")

