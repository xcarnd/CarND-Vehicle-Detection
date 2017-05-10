# -*- encoding: utf-8 -*-

import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import cv2


def get_hog(image, orientations=9, pixels_per_cell=8, cells_per_block=2, channel=-1, visualise=False,
            feature_vector=False):
    """Extract HOG features for the specified image.
    
    By specifying channel as -1, HOG features extraction will be applied to each channel of the image and 
    then concatenated as the final feature vector.
    """
    if channel == -1:
        hog_features = []
        for channel in range(image.shape[2]):
            hog_channel = hog(image[:, :, channel], orientations=orientations,
                              pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                              cells_per_block=(cells_per_block, cells_per_block), visualise=False,
                              feature_vector=False)
            hog_features.append(hog_channel)
        return np.stack(hog_features, axis=-1)
    else:
        img_plane = image[:, :, channel]
        return hog(img_plane, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(cells_per_block, cells_per_block), visualise=visualise,
                   feature_vector=feature_vector)


def get_color_hist(image, nbins=32):
    """Extract color histogram features for the specified image.
    """
    hist_0, _ = np.histogram(image[:, :, 0], bins=nbins)
    hist_1, _ = np.histogram(image[:, :, 1], bins=nbins)
    hist_2, _ = np.histogram(image[:, :, 2], bins=nbins)

    return np.concatenate((hist_0, hist_1, hist_2))


def get_spatial_binning(image, size=(32, 32)):
    """Extract spatial binning features for the specified image."""
    channels = []
    for ch in range(image.shape[2]):
        channels.append(cv2.resize(image[:, :, ch], size))
    return np.concatenate(channels)


def get_feature_vector(image,
                       subsampled_hog_features=None,
                       hog_orientations=9,
                       hog_pixels_per_cell=8,
                       hog_cells_per_block=2,
                       hog_channel=-1):
    """Returns features of image as a 1D vector.
    """
    spatial_binning_features = get_spatial_binning(image, size=(4, 4))
    hist_features = get_color_hist(image)
    if subsampled_hog_features is None:
        hog_features = get_hog(image,
                               orientations=hog_orientations,
                               pixels_per_cell=hog_pixels_per_cell,
                               cells_per_block=hog_cells_per_block,
                               channel=hog_channel)
    else:
        hog_features = subsampled_hog_features

    result = np.concatenate((spatial_binning_features.ravel(), hist_features.ravel(), hog_features.ravel(),))
    # result = np.concatenate((hog_features.ravel(), ))
    return result


def get_feature_normalizer(fitting_features):
    """Returns a feature vector normalizer by fitting against the provided features.
    """
    scaler = StandardScaler()
    scaler.fit(fitting_features)
    return scaler
