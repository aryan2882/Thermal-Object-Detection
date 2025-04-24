import cv2
import numpy as np

def normalize_pixels(image):
    """
    Normalize pixel values to the range [0,1] (float32 format).

    Parameters:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: Normalized image.
    """
    return image.astype(np.float32) / 255.0  # Scale pixel values between 0 and 1
