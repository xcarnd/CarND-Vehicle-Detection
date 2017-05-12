# -*- encoding: utf-8 -*-

import numpy as np
from skimage.feature import hog
import cv2
import settings as s


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
        return np.stack(hog_features)
    else:
        img_plane = image[:, :, channel]
        return hog(img_plane, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(cells_per_block, cells_per_block), visualise=visualise,
                   feature_vector=feature_vector)


def get_color_hist(image, nbins=32):
    """Extract color histogram features for the specified image.
    """
    channels = []
    for channel in range(image.shape[2]):
        channels.append(np.histogram(image[:, :, channel], bins=nbins)[0])
    return np.stack(channels)


def get_spatial_binning(image, size=(32, 32)):
    """Extract spatial binning features for the specified image.
    """
    channels = []
    for channel in range(image.shape[2]):
        channels.append(cv2.resize(image[:, :, channel], size))
    return np.stack(channels)


def get_feature_vector(image,
                       subsampled_hog_features=None):
    """Returns features of image as a 1D vector.

    Subsampled HOG features are used directly if provided.
    """
    spatial_binning_features = get_spatial_binning(image, size=s.binning_size)
    hist_features = get_color_hist(image, nbins=s.hist_bins)
    if subsampled_hog_features is None:
        hog_features = get_hog(image,
                               orientations=s.hog_orientations,
                               pixels_per_cell=s.hog_pixels_per_cell,
                               cells_per_block=s.hog_cells_per_block,
                               channel=s.hog_channel)
    else:
        hog_features = subsampled_hog_features

    feats = []
    if s.include_binning:
        feats.append(spatial_binning_features.ravel())
    if s.include_hist:
        feats.append(hist_features.ravel())
    if s.include_hog:
        feats.append(hog_features.ravel())

    result = np.concatenate(feats)
    return result
