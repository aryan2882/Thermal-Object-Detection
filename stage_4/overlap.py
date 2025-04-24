import cv2
import numpy as np

def remove_overlap(image):
    """
    Remove overlapping objects in an image using morphological operations and contour filtering.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.

    Returns:
        numpy.ndarray: Image with overlapping objects removed.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to detect objects
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to separate objects
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask to draw non-overlapping contours
    mask = np.zeros_like(gray)

    for cnt in contours:
        # Get bounding rectangle of contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Check if the object is overlapping (based on area)
        if w * h > 500:  # Adjust this threshold as needed
            continue  # Skip overlapping large regions
        
        cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

    # Apply the mask to remove overlapping objects
    result = cv2.bitwise_and(image, image, mask=mask)

    return result
