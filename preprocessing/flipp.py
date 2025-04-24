import cv2

def flip_image(image, flip_code=1):
    """
    Flip an image horizontally, vertically, or both.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        flip_code (int): Flip mode - 0 (vertical), 1 (horizontal), -1 (both).

    Returns:
        numpy.ndarray: Flipped image.
    """
    return cv2.flip(image, flip_code)
