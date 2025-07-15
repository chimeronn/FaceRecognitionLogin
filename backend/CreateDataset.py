import numpy as np
from glob import glob
from PIL import Image
import math

def create_dataset(folder, isFace):
    images = []
    test_images = []
    files = sorted(glob(f"{folder}/*.jpg"))
    labels = [isFace] * math.ceil(0.9 * len(files))
    test_labels = [isFace] * (len(files) - len(labels))
    i = 0
    for filename in files:
        img = Image.open(filename).convert("L")
        img = np.asarray(img).flatten() / 255.0
        if i >= 0.9 * len(files):
            test_images.append(img)
        else:
            images.append(img)
            i += 1
    return np.array(images), np.array(test_images), labels, test_labels