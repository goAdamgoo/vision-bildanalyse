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
plt.savefig(os.path.join('debug_output', '02_blur.png'), bbox_inches='tight')
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
plt.savefig(os.path.join('debug_output', '03_mask.png'), bbox_inches='tight')
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
# save edges
plt.savefig(os.path.join('debug_output', '05_edges.png'), bbox_inches='tight')
plt.close()

# ------------------------------------------------
# 7. Contour detection (OpenCV) on cleaned mask
# Konturenerkennung mit OpenCV
contours, _ = cv2.findContours(
    mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
# filter contours by area
filtered_contours = [c for c in contours if cv2.contourArea(c) > 800]
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_contours, filtered_contours, -1, (0, 255, 0), 1)

plt.figure(figsize=(6, 4))
plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title(f'Filtered contours: {len(filtered_contours)}')
plt.axis('off')
plt.savefig(os.path.join('debug_output', '07_contours.png'), bbox_inches='tight')
plt.show()
plt.close()
