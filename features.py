# -*- encoding: utf-8 -*-

import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import utils


def get_color_hist(img, nbins=32, bins_range=(0, 256)):
    """Extract histogram features from `img`. `img` must be image with 3 channels (RGB, HLS, HSV, etc.)
    """
    # Compute the histogram of the color channels
    # the 2nd return value contains the bin edges which are not used
    channel_0_hist, _ = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel_1_hist, _ = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel_2_hist, _ = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # combine all the histograms into a single feature vector
    return np.concatenate((channel_0_hist, channel_1_hist, channel_2_hist))


def get_hog(img, orientations=9, pixels_per_cell=8, cells_per_block=2, channel=-1, visualise=False,
            feature_vector=False):
    """Extract HOG features for the specified image.
    
    By specifying channel as -1, HOG features extraction will be applied to each channel of the image and 
    then concatenated as the final feature vector.
    """
    if channel == -1:
        hog_channel_0 = hog(img[:, :, 0], orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                            cells_per_block=(cells_per_block, cells_per_block), visualise=False,
                            feature_vector=False)
        hog_channel_1 = hog(img[:, :, 1], orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                            cells_per_block=(cells_per_block, cells_per_block), visualise=False,
                            feature_vector=False)
        hog_channel_2 = hog(img[:, :, 2], orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                            cells_per_block=(cells_per_block, cells_per_block), visualise=visualise,
                            feature_vector=False)
        return np.stack((hog_channel_0, hog_channel_1, hog_channel_2), axis=-1)
    else:
        img_plane = img[:, :, channel]
        return hog(img_plane, orientations=orientations, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                   cells_per_block=(cells_per_block, cells_per_block), visualise=visualise,
                   feature_vector=feature_vector)


def get_feature_vector(img, color_space='RGB'):
    """Returns features of image as a 1D vector.
    """
    image = utils.convert_color_space(img, color_space)

    hog_features = get_hog(image)
    result = np.concatenate((hog_features.ravel(),))

    return result


def get_feature_normalizer(fitting_features):
    """Returns a feature vector normalizer by fitting against the provided features.
    """
    scaler = StandardScaler()
    scaler.fit(fitting_features)
    return scaler


def normalize_features(scaler, features):
    """Normalize given features.
    """
    print("Normalizing features")
    return scaler.transform(features)
