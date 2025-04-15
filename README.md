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


