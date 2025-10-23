"""
Author: Vikram Sandu
Date: 2025-09-25
Description:
    This script creates various augmentations on the elements
    before pasting them on the sheets.
"""
import cv2
import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance


# Aug-1
def rand_scale_(element):
    scale = random.uniform(0.3, 0.8)
    w, h = element.size
    element = element.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return element


# Aug-2
def hflip_(element, prob=0.4):
    # Horizontal flip
    if random.random() < prob:
        element = ImageOps.mirror(element)
    return element


# Aug-3
def rand_rotate_(element, max_degree=1):
    # Random rotation
    angle = random.uniform(-max_degree, max_degree)
    element = element.rotate(angle, expand=True)
    return element


# Aug-4
def enhance_(element, prob=0.25):
    if random.random() < prob:
        enhancer_b = ImageEnhance.Brightness(element)
        element = enhancer_b.enhance(random.uniform(0.75, 1.25))

    if random.random() < prob:
        enhancer_c = ImageEnhance.Contrast(element)
        element = enhancer_c.enhance(random.uniform(0.75, 1.25))

    return element


# Aug-5
def morph_(element, prob=0.1):
    if random.random() < prob:
        element_cv = cv2.cvtColor(np.array(element), cv2.COLOR_RGBA2BGRA)
        alpha = element_cv[:, :, 3]
        kernel_size = random.choice([1, 2])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() > 0.5:
            alpha = cv2.dilate(alpha, kernel, iterations=1)
        else:
            alpha = cv2.erode(alpha, kernel, iterations=1)
        element_cv[:, :, 3] = alpha
        element = Image.fromarray(cv2.cvtColor(element_cv, cv2.COLOR_BGRA2RGBA))

    return element


def augment_(element):
    return (
        morph_(
            enhance_(
                rand_rotate_(
                    hflip_(
                        rand_scale_(element)
                    )
                )
            )
        ))


if __name__ == "__main__":
    pass
