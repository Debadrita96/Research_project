Comparative Evaluation of Transformer-Based Detectors for Coronary Stenosis Localization

This research project investigates the effectiveness of modern transformer-based object detection architectures for detecting coronary artery stenosis in X-ray angiography using the ARCADE (MICCAI 2023) dataset.

Developed at the Pattern Recognition Lab, FAU Erlangen-Nürnberg.

Research Motivation

Detecting coronary stenosis presents unique challenges:

Extremely small lesion size

Low contrast boundaries

High anatomical variability

Severe class imbalance

Limited labeled training data

Transformer detectors have demonstrated strong performance on natural images, but their behavior on subtle, fine-grained medical anomalies remains underexplored.

This project evaluates whether transformer architectures can reliably localize stenotic plaques under such constraints.

Models Evaluated

Three transformer-based object detectors were systematically compared:

Conditional DETR (ResNet-50 backbone)

Grounding DINO (Swin-L backbone)

Grounding DINO (PVT backbone variant)

All models were fine-tuned under identical training protocols for fair benchmarking.

Framework & Implementation

This project was implemented using:

PyTorch

MMDetection 3.3.0

MMCV

Core detection architectures were adapted from the official OpenMMLab MMDetection repository (Apache 2.0 License):

https://github.com/open-mmlab/mmdetection

What Was Adapted

Model architecture implementations (Conditional DETR, DINO variants)

COCO evaluation API

Data pipeline structure

Training loops and checkpointing

COCO error analysis utilities

What Was Specifically Implemented and Investigated

ARCADE dataset validation and COCO-format verification

Severe class imbalance analysis (26 classes, dominant stenosis class)

Controlled backbone ablation (Swin-L vs PVT)

Multi-scale training configuration

Hyperparameter tuning (epochs, LR scheduling, batch size)

Small-object detection evaluation (AP_small)

Confidence score distribution analysis

COCO error decomposition (FN, BG, localization errors)

Qualitative failure-mode inspection

Literature-aligned performance comparison

This project goes beyond running default configs by critically analyzing architecture behavior under medical constraints.

Experimental Setup

Dataset: ARCADE Stenosis Dataset (MICCAI 2023)

~1,500 X-ray angiography images

COCO-format bounding box annotations

Highly imbalanced class distribution

Training Details:

Conditional DETR fine-tuned from COCO-pretrained weights

Grounding DINO (Swin-L) fine-tuned from open-vocabulary pretrained weights

Grounding DINO (PVT) initialized from COCO-pretrained DINO

Multi-scale augmentation used for DINO variants

Early stopping based on validation mAP

Evaluation Metrics:

mAP@[0.5:0.95]

AP_small

Average Recall (AR)

COCO error breakdown

Quantitative Results
Model	mAP@[0.5:0.95]	AP_small	AR
Conditional DETR	0.000	0.016	0.017
Grounding DINO (Swin-L)	0.153	0.332	0.397
Grounding DINO (PVT)	0.018	0.080	0.228
Key Technical Insights
1️⃣ Vanilla DETR struggles with subtle lesions

Conditional DETR failed to learn meaningful localization (mAP ≈ 0), confirming that global attention without strong multi-scale representation is insufficient for tiny medical abnormalities.

2️⃣ Backbone strength is critical

Grounding DINO with Swin-L significantly outperformed both other models:

0.153 mAP (nearly 2x baseline transformer reports in literature)

0.332 AP_small

0.397 AR

This highlights:

Importance of hierarchical feature extraction

Value of large-scale pretraining

Effectiveness of denoising queries and multi-scale features

3️⃣ Lightweight transformer backbones underperform

Replacing Swin-L with PVT drastically reduced performance.
Despite multi-scale design, PVT lacked sufficient representational power to distinguish subtle stenotic patterns from background noise.

4️⃣ Error Analysis Findings

COCO error decomposition revealed:

Conditional DETR → Dominant background confusion and false negatives

Swin-L → Good small-object recall but localization precision remains challenging

PVT → High false negative rate, weak confidence discrimination

This reinforces that small-object medical detection demands strong spatial inductive bias and hierarchical features.

Deep Learning Contributions

This project demonstrates:

Practical deployment of large-scale transformer detection frameworks

Architectural sensitivity analysis in low-data medical settings

Backbone ablation under clinical constraints

Quantitative + qualitative detection error profiling

Analysis of transformer domain gap (COCO → angiography)

Why This Matters

Transformer detectors do not automatically generalize to medical imaging.

This study provides empirical evidence that:

Pretraining alone is insufficient

Backbone capacity dominates performance

Multi-scale feature design is essential

Domain-specific adaptation is required for clinical viability

Attribution

This project builds upon the OpenMMLab ecosystem:

MMDetection

MMCV

Conditional DETR implementation

Grounding DINO integration

All core architecture implementations belong to their original authors.
This repository focuses on controlled experimentation, adaptation, and analysis within a medical imaging context.

