from skimage import data
import matplotlib.pyplot as plt

# Beépített tesztkép (érmék)
img = data.coins()

plt.imshow(img, cmap="gray")
plt.title("Eredeti kép")
plt.axis("off")
plt.show()
