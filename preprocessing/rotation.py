import cv2

def rotate_image(image, angle):
    """
    Rotate an image by a given angle.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        angle (float): Rotation angle in degrees (counterclockwise).

    Returns:
        numpy.ndarray: Rotated image.
    """
    (h, w) = image.shape[:2]  # Get image dimensions
    center = (w // 2, h // 2)  # Get center of image

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image
