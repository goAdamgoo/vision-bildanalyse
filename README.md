# Computer Vision Demo
*A modular image preprocessing pipeline for computer vision*

**Author:** Ádám Bege  
**Languages:** Python 3 · OpenCV · scikit-image · NumPy · Matplotlib  

---

## Overview / Überblick

**EN:**  
This project demonstrates a complete image preprocessing pipeline for computer vision applications such as **object segmentation** and **geometric feature extraction**.  
It combines Gaussian smoothing, adaptive thresholding, morphological filtering, edge detection, and shape-based contour analysis using `skimage.data.coins()` as a sample dataset.

**DE:**  
Dieses Projekt zeigt eine vollständige Bildvorverarbeitungspipeline für Computer-Vision-Anwendungen, z. B. **Objektsegmentierung** oder **Merkmalextraktion**.  
Das Skript kombiniert Gauß-Filterung, adaptive Schwellenwertbildung, morphologische Filterung, Kantenerkennung und Formanalyse anhand des Beispiels *coins*.

---

## Features
- Gaussian noise reduction  
- Adaptive (local) thresholding  
- Morphological cleaning (opening & closing)  
- Edge detection (Canny)  
- Contour extraction and shape-based filtering  
- Fully modular Python function (`detect_objects()`)  
- Notebook-compatible (Jupyter / VS Code)  

---

## Requirements / Voraussetzungen

**Python Version:** 3.10 or later  

**Dependencies:**
numpy
opencv-python
scikit-image
matplotlib


Install via:
```bash
pip install -r requirements.txt

Usage

Run the demo directly in a Jupyter Notebook or Python environment:

from vision_demo import detect_objects

results = detect_objects()

Returned data:
dict_keys([
  'original', 'blur', 'local_thresh', 'mask',
  'mask_clean', 'edges', 'img_contours',
  'img_boxes', 'contours_info'
])

Visualizations appear inline in Jupyter.
