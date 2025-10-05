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
# 2. Gaussian blur (noise reduction)
# Gauß-Filter (Rauschunterdrückung)
blur = cv2.GaussianBlur(img, (5, 5), 0)
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
plt.title(f'Otsu threshold (t = {thresh_val:.2f})')
plt.axis('off')
plt.savefig(os.path.join('debug_output', '03_mask.png'), bbox_inches='tight')
plt.close()

# ------------------------------------------------
# 4. Edge detection (Canny)
#Kantendetektion (Canny)
edges = feature.canny(mask, sigma=1.0)
plt.figure(figsize=(6, 4))
plt.imshow(edges, cmap='gray')
plt.title('Canny edge detection')
plt.axis('off')
# save edges
plt.savefig(os.path.join('debug_output', '04_edges.png'), bbox_inches='tight')
plt.close()

# ------------------------------------------------
# 5. Contour detection (OpenCV)
# Konturenerkennung mit OpenCV
contours, _ = cv2.findContours(
    (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)

plt.figure(figsize=(6, 4))
plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title(f'Contours detected: {len(contours)}')
plt.axis('off')
plt.savefig(os.path.join('debug_output', '05_contours.png'), bbox_inches='tight')
plt.show()
plt.close()
