import numpy as np
def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    image = np.array(image)
    gray_image = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    return gray_image.tolist()