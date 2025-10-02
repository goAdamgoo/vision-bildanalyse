from skimage import data
import matplotlib.pyplot as plt

img = data.coins()

plt.imshow(img, cmap="gray")
plt.title("Eredeti kép")
plt.axis("off")
plt.show()

import cv2  # OpenCV kell hozzá

# Gaussian Blur alkalmazása
blur = cv2.GaussianBlur(img, (5,5), 0)

# Eredmény összehasonlítása
import matplotlib.pyplot as plt

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Eredeti")

plt.subplot(1,2,2)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")

plt.show()

from skimage import filters

# Automatikus küszöbérték Otsu módszerrel
thresh_val = filters.threshold_otsu(blur)
mask = blur > thresh_val

# Eredmény megjelenítése
plt.subplot(1,2,1)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title(f"Otsu threshold (t={thresh_val:.2f})")

plt.show()
