from typing import Tuple

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature import canny

# Demo list of car models to select from deterministically
CAR_MODELS = [
    "Toyota Camry",
    "Honda Civic",
    "Ford F-150",
    "BMW 3 Series",
    "Audi Q5",
]


def preprocess_tire_print(image: Image.Image) -> Image.Image:
    """Return a visualization-friendly processed image (edges/contrast)."""
    arr = np.array(image).astype(np.float32) / 255.0
    gray = rgb2gray(arr)

    # Combine sobel and canny for a crisp preview
    edges_canny = canny(gray, sigma=1.2)
    edges_sobel = sobel(gray)

    edges_mix = np.clip(edges_sobel * 0.6 + edges_canny.astype(np.float32) * 0.8, 0, 1)
    edges_img = (edges_mix * 255).astype(np.uint8)
    return Image.fromarray(edges_img)


def _feature_signature(image: Image.Image) -> Tuple[float, float, float]:
    """A simple feature signature based on pixel stats (deterministic)."""
    arr = np.array(image.convert("L"), dtype=np.float32)
    # Normalize
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    # Downsample for speed
    small = arr[::8, ::8]
    mean = float(np.mean(small))
    std = float(np.std(small))
    skew = float(np.mean(((small - mean) / (std + 1e-6)) ** 3))
    return mean, std, skew


def predict_car_model(image: Image.Image) -> Tuple[str, float]:
    """Mock predictor returning a car model name and confidence.

    Deterministic mapping using simple image statistics, so the same image
    yields the same output. Replace this logic with a real model when available.
    """
    mean, std, skew = _feature_signature(image)

    # Deterministic index selection
    idx = int(abs(mean * 7 + std * 11 + skew * 13)) % len(CAR_MODELS)
    model = CAR_MODELS[idx]

    # Confidence as a bounded function of variance and edge content
    # Higher std generally implies clearer tread patterns
    conf = 1 / (1 + np.exp(-std * 2.0))  # sigmoid in [0,1]
    conf = float(np.clip(conf, 0.15, 0.97))

    return model, conf
