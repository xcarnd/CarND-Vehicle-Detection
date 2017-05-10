# -*- encoding: utf-8 -*-
import cv2


def convert_color_space(image, color_space):
    """Utility function for converting between BGR and another color spaces.
    """
    if color_space == 'RGB':
        result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == 'HLS':
        result = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    elif color_space == 'HSV':
        result = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        raise Exception("{} not supported yet.".format(color_space))
    return result
