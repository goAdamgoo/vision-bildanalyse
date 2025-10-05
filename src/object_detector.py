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

from skimage import data, filters, feature, color
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------
# 1. Load grayscale image (sample: coins)
# Graustufenbild laden (Beispiel: "coins")
img = data.coins()
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.axis('off')

# ------------------------------------------------
# 2. Gaussian blur (noise reduction)
# Gauß-Filter (Rauschunterdrückung)
blur = cv2.GaussianBlur(img, (5, 5), 0)
plt.subplot(1, 2, 2)
plt.imshow(blur, cmap='gray')
plt.title('Gaussian blur')
plt.axis('off')
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 3. Automatic thresholding (Otsu method)
# Automatische Schwellenwertbestimmung (Otsu-Verfahren)
thresh_val = filters.threshold_otsu(blur)
mask = blur > thresh_val

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(blur, cmap='gray')
plt.title('Blurred image')

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title(f'Otsu threshold (t = {thresh_val:.2f})')
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 4. Edge detection (Canny)
#Kantendetektion (Canny)
edges = feature.canny(mask, sigma=1.0)
plt.figure(figsize=(6, 4))
plt.imshow(edges, cmap='gray')
plt.title('Canny edge detection')
plt.axis('off')
plt.show()

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
plt.show()
