"""
Computer Vision Demo
Author: Ádám Bege

Demonstrates a complete image preprocessing pipeline including:
- Gaussian smoothing
- Adaptive thresholding
- Morphological cleaning
- Edge and contour detection
- Shape-based contour filtering

Dieses Skript demonstriert eine einfache Bildvorverarbeitungspipeline:
Glättung, adaptive Schwellenwertbildung, morphologische Reinigung,
Kantenerkennung und Formanalyse anhand des Beispiels 'coins'.
"""

from skimage import data, filters, feature
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, Tuple


def detect_objects(
    img: np.ndarray = None,
    debug_dir: str = "debug_output",
    block_size: int = 51,
    morph_kernel: Tuple[int, int] = (5, 5),
    circ_min: float = 0.80,
    solidity_min: float = 0.85,
    fit_ratio_min: float = 0.80,
) -> Dict[str, Any]:
    """Run the object detection pipeline and return intermediate images.

    Returns a dict with keys:
      - original: original grayscale image
      - blur: blurred image
      - local_thresh: local threshold image (float)
      - mask: binary mask (bool)
      - mask_clean: cleaned uint8 mask
      - edges: edge image (bool)
      - img_contours: image with all filtered contours drawn (BGR)
      - img_boxes: final annotated image (BGR)
      - contours_info: list of per-contour metrics
    """
    os.makedirs(debug_dir, exist_ok=True)

    # Load sample image if none provided
    if img is None:
        img = data.coins()

    result: Dict[str, Any] = {}
    result["original"] = img.copy()

    # 1. Blur
    blur = cv2.GaussianBlur(img, (9, 9), 2)
    result["blur"] = blur

    # 2. Local adaptive threshold
    local_thresh = filters.threshold_local(blur, block_size, method="mean")
    mask = blur > local_thresh
    result["local_thresh"] = local_thresh
    result["mask"] = mask

    # 3. Morphological cleaning
    kernel = np.ones(morph_kernel, np.uint8)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_open = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    result["mask_clean"] = mask_clean
    plt.imsave(os.path.join(debug_dir, "04_mask_clean.png"), mask_clean, cmap="gray")

    # 4. Edges
    edges = feature.canny(mask_clean > 0, sigma=1.0)
    result["edges"] = edges

    # 5. Contours and filtering
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    median_area = np.median(areas) if len(areas) > 0 else 800
    min_area = max(200, int(median_area * 0.5))
    result["min_area"] = min_area

    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contours, filtered_contours, -1, (0, 255, 0), 1)
    result["img_contours"] = img_contours

    # 6. Shape-based filtering
    accepted_contours = []
    contours_info = []
    for c in filtered_contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        fit_area = np.pi * (radius ** 2)
        fit_ratio = area / fit_area if fit_area > 0 else 0

        info = {
            "area": float(area),
            "perimeter": float(perimeter),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "fit_ratio": float(fit_ratio),
            "centroid": (float(cx), float(cy)),
        }
        contours_info.append(info)

        if (circularity >= circ_min and solidity >= solidity_min and fit_ratio >= fit_ratio_min):
            accepted_contours.append(c)

    result["contours_info"] = contours_info

    # 7. Visualization: draw final boxes
    img_boxes = img_contours.copy()
    for idx, c in enumerate(accepted_contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(img_boxes, str(idx), (x + 2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    result["img_boxes"] = img_boxes
    # Save final annotated image for convenience
    plt.imsave(os.path.join(debug_dir, "07_contours_boxes.png"), cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))

    return result


if __name__ == "__main__":
    # Minimal demo: run pipeline and save results 
    res = detect_objects()
    print(f"Detected {len(res.get('img_boxes', []))} pixels in final image (see debug_output)")
