# ============================================
# HD Map & Computer Vision Demo
# Author: Ádám Bege
# Goal: Demonstrate image preprocessing pipeline
# 
# Dieses Skript demonstriert eine einfache Bildvorverarbeitungspipeline
# (Glättung, Schwellenwertbildung, Kantendetektion, Konturfindung) anhand
# des Beispielbildes `skimage.data.coins()`.
# Autor: Ádám Bege
# ============================================

from skimage import data, filters, feature, color, exposure
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------
# 1. Load grayscale image (sample: coins)
# Graustufenbild laden (Beispiel: "coins")
img = data.coins()
# ensure debug output directory exists
os.makedirs('debug_output', exist_ok=True)

# ------------------------------------------------
# 1. Original image
plt.figure(figsize=(6, 4))
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.axis('off')
plt.savefig(os.path.join('debug_output', '01_original.png'), bbox_inches='tight')
plt.close()

# ------------------------------------------------
# 2. Stronger Gaussian blur (noise reduction)
# Stärkerer Gauß-Filter (Rauschunterdrückung)
blur = cv2.GaussianBlur(img, (9, 9), 2)
plt.figure(figsize=(6, 4))
plt.imshow(blur, cmap='gray')
plt.title('Gaussian blur')
plt.axis('off')
# blur image not saved as debug output to keep only real results
plt.close()

# ------------------------------------------------
# 3. Adaptive/local thresholding
# Lokale Schwellenwertbildung (adaptiv)
# Use a block size appropriate for coin size; odd integer
block_size = 51
local_thresh = filters.threshold_local(blur, block_size, method='mean')
mask = blur > local_thresh
thresh_val = None

# ------------------------------------------------
# 3. Threshold mask
plt.figure(figsize=(6, 4))
plt.imshow(mask, cmap='gray')
plt.axis('off')
# raw adaptive mask not saved as debug output (keeping only cleaned mask)
plt.close()

# ------------------------------------------------
# 5. Morphological cleaning (open then close) and contour filtering
# Morphologische Reinigung (Öffnen dann Schließen) und Konturfilterung
kernel = np.ones((5, 5), np.uint8)
# cv2 morphology expects uint8 0/255
mask_u8 = (mask.astype(np.uint8) * 255)
mask_open = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
# Save cleaned mask
plt.figure(figsize=(6, 4))
plt.imshow(mask_clean, cmap='gray')
plt.title('Mask after morphological cleaning')
plt.axis('off')
plt.savefig(os.path.join('debug_output', '04_mask_clean.png'), bbox_inches='tight')
plt.close()

# ------------------------------------------------
# 6. Edge detection (Canny) on cleaned mask
# Kantendetektion (Canny)
edges = feature.canny(mask_clean > 0, sigma=1.0)
plt.figure(figsize=(6, 4))
plt.imshow(edges, cmap='gray')
plt.title('Canny edge detection')
plt.axis('off')
# edges not saved to debug output
plt.close()

# ------------------------------------------------
# 7. Contour detection (OpenCV) on cleaned mask
# Konturenerkennung mit OpenCV
contours, _ = cv2.findContours(
    mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
# compute automatic min-area heuristic from contour areas
areas = [cv2.contourArea(c) for c in contours]
if len(areas) > 0:
    median_area = float(np.median(areas))
    min_area = max(200, int(median_area * 0.5))
else:
    min_area = 800

# filter contours by automatic min_area
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_contours, filtered_contours, -1, (0, 255, 0), 1)

# ------------------------------------------------
# Compute shape metrics and filter by circularity/solidity/fit_ratio
# Defaults: circularity>0.80, solidity>0.85, fit_ratio>0.80
circ_min = 0.80
solidity_min = 0.85
fit_ratio_min = 0.80

accepted_contours = []
metrics = []
for c in filtered_contours:
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = 0.0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)

    # solidity via convex hull
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # min enclosing circle fit ratio
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    fit_area = np.pi * (radius ** 2)
    fit_ratio = area / fit_area if fit_area > 0 else 0

    metrics.append({'area': area, 'circularity': circularity, 'solidity': solidity, 'fit_ratio': fit_ratio})

    if circularity >= circ_min and solidity >= solidity_min and fit_ratio >= fit_ratio_min:
        accepted_contours.append(c)

# Draw bounding boxes and numeric labels for each accepted contour
img_boxes = img_contours.copy()
for idx, c in enumerate(accepted_contours, start=1):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.putText(img_boxes, str(idx), (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
plt.figure(figsize=(6, 4))
plt.imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
plt.title(f'Accepted contours: {len(accepted_contours)} (min_area={min_area})')
plt.axis('off')
plt.savefig(os.path.join('debug_output', '07_contours_boxes.png'), bbox_inches='tight')
plt.close()

# Optionally, if you want the rejected ones too, you can inspect metrics list
