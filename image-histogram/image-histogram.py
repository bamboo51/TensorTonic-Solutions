import numpy as np
def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    image = np.array(image, dtype=np.uint8)
    return list(np.bincount(image.ravel(), minlength=256))