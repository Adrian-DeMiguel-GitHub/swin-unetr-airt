# Swin UNETR for Spatio-Temporal Semantic Segmentation in Active Infrared Thermography

This repository contains the implementation of the codebase developed for the master's thesis:

**_Spatiotemporal Transformer Approach for Semantic Segmentation in Internal Defect Detection using Active Infrared Thermography_**

**Author**: Adrián de Miguel Palacio  
**Degree**: Master of Science (Data Science)  
**Institution**: Institut für Informatik und Computational Science, Universität Potsdam, Germany  
**Year**: 2025

---

## 🧠 Overview

This thesis introduces and evaluates a novel end-to-end spatio-temporal deep learning framework for detecting internal defects using **Active Infrared Thermography (AIRT)**. The proposed method is based on a **3D transformer architecture (Swin UNETR)**, adapted to process thermal image sequences directly without relying on traditional video-to-image preprocessing techniques.

The codebase also includes a benchmark against a common 2D spatial deep learning approach: **U-Net with a VGG11 backbone**, using **Principal Component Thermography (PCT)** and **Pulsed Phase Thermography (PPT)** for video-to-image preprocessing.

---

## 🧩 Models

### 🔷 Proposed Approach: Swin UNETR

- Fully end-to-end 3D spatio-temporal transformer-based segmentation
- Tiled spatial inputs + temporal subsampling
- Two variants tested:
  - **V1**: 3D logit compression at the output
  - **V2**: Temporal compression integrated in skip connections

### 🔶 Baseline: U-Net VGG11

- 2D segmentation on summary images created with video-to-image processing
- Evaluated using both **PPT** and **PCT**

---

## 🧪 Evaluation Protocol

- Hold-out validation + 5-fold cross-evaluation for robust performance estimation

- Metrics (@0.5 wehn evaluated at a fixed classification threshold of 0.5):
  - **IoU@0.5 (Intersection over Union)**: Measures the overlap between predicted and true defect regions
  - **F1-Score@0.5 (Dice Coefficient)**: Harmonic mean of precision and recall for defective areas
  - **TPR@0.5 (True Positive Rate / Recall)**: Percentage of actual defective pixels correctly identified  
  - **FPR@0.5 (False Positive Rate)**: Percentage of non-defective pixels incorrectly classified as defective  
  - **ROC AUC**: Area under the ROC curve across varying classification thresholds

- Statistical testing for significance:
  - **Shapiro–Wilk test** to assess normality of metric differences
  - **Paired one-tailed t-tests** to compare performance between models

- Complexity metrics:
  - **FLOPs**: Approximate computational cost
  - **# Parameters**: Model size
  - **Training time / Inference time**: Runtime performance evaluation



