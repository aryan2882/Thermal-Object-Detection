import cv2
import numpy as np

def enhance_contrast(image, method="clahe"):
    """
    Enhance the contrast of an image using either CLAHE or Histogram Equalization.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        method (str): Contrast enhancement method, either "clahe" or "hist_eq".

    Returns:
        numpy.ndarray: Image with enhanced contrast.
    """
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "hist_eq":
        # Apply standard histogram equalization
        enhanced_image = cv2.equalizeHist(gray)
    elif method == "clahe":
        # Create CLAHE object (Clip Limit helps in avoiding over-amplification of noise)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)
    else:
        raise ValueError("Invalid method. Use 'clahe' or 'hist_eq'.")

    # Convert single-channel image back to 3-channel by merging with itself
    enhanced_image = cv2.merge([enhanced_image] * 3)

    return enhanced_image

# Example usage:
# image = cv2.imread("input.jpg")
# enhanced = enhance_contrast(image, method="clahe")
# cv2.imshow("Enhanced Image", enhanced)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
