
import cv2
import numpy as np

IMG_SIZE = 48
# i resize every face to this fixed size so model input size stays same


def contrast_normalize(img):
    """Apply basic contrast change with CLAHE."""
    # i use clahe here so i improve local contrast in face images
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def grayscale_verify(img):
    """Make sure image is gray."""
    # i make sure image is single channel gray before further processing
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def preprocess_image(img, use_clahe=True):
    """Preprocess one image for the model."""
    # i convert to gray resize apply optional clahe and scale to zero one then flatten
    img = grayscale_verify(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    if use_clahe:
        img = contrast_normalize(img)
    img = img.astype(np.float64) / 255.0
    return img.flatten()


def preprocess_batch(images, use_clahe=True):
    """Preprocess many images in a loop."""
    # i loop over full batch and call same preprocess for each image so pipeline is simple
    out = []
    for i in range(images.shape[0]):
        out.append(preprocess_image(images[i], use_clahe=use_clahe))
    return np.array(out, dtype=np.float64)
