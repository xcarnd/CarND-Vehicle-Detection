# -*- encoding: utf-8 -*-

import classifier
import cv2
import utils
import features as f
import matplotlib.pyplot as plt
import numpy as np


class Pipeline(object):
    def __init__(self, classifier):
        self.classifier = classifier

    def search_for_matches(self, image, region_of_interest=None, scale=1.0, visualize=False, color_space='YCrCb'):
        """Apply sliding window search on the given image.
        
        :param image: the region which search is imposed on.
         
        :param region_of_interest: region in which the search is limited in. If unspecified (None), defaults to the  
            full region of the image. Specified in the format:
            `((top-left-x, top-left-y), (bottom-right-x, bottom-right-y))`
        
        :param scale: Searching window scales.

        :param color_space: Color space used for searching

        :param visualize: If True, returns a visualizing image.
        """
        if visualize:
            # note: format for visualize_img is BGR
            visualize_img = np.copy(image)

        if region_of_interest is None:
            region_of_interest = ((0, 0), (image.shape[1], image.shape[0]))

        x_start, x_stop = region_of_interest[0][0], region_of_interest[1][0]
        y_start, y_stop = region_of_interest[0][1], region_of_interest[1][1]

        search_region = image[y_start:y_stop, x_start:x_stop, :]
        search_region = utils.convert_color_space(search_region, color_space)
        print("Shape of search region: ", search_region.shape)

        # scaling the input if necessary
        if scale != 1:
            search_region = cv2.resize(search_region,
                                       (int(search_region.shape[1] / scale), int(search_region.shape[0] / scale)))
            print("Scaled shape of search region: ", search_region.shape)

        # parameters tuning:
        # number of pixels per cell when extracting HOG features
        pixels_per_cell = 8
        # size (number of pixels) of window
        size_window = 64
        # number of cells per HOG feature extraction block
        cells_per_block = 2

        # number of blocks per sliding window
        blocks_per_window = (size_window // pixels_per_cell) - cells_per_block + 1
        # cell increments for sliding
        inc_cells = 2

        # number of (complete) blocks horizontally (along x) / vertically (along y)
        num_blocks_x = (search_region.shape[1] // pixels_per_cell) - cells_per_block + 1
        num_blocks_y = (search_region.shape[0] // pixels_per_cell) - cells_per_block + 1
        # number of windows horizontally (along x) / vertically (along y)
        stepx = (num_blocks_x - blocks_per_window) // inc_cells + 1
        stepy = (num_blocks_y - blocks_per_window) // inc_cells + 1

        # get HOG features for the whole search region
        hog_features = f.get_hog(search_region,
                                 pixels_per_cell=pixels_per_cell,
                                 cells_per_block=cells_per_block,
                                 channel=-1)

        # result window rects
        rects = []

        for x in range(stepx):
            for y in range(stepy):
                xpos = x * inc_cells
                ypos = y * inc_cells
                x_tl = xpos * pixels_per_cell
                y_tl = ypos * pixels_per_cell
                win_img = search_region[y_tl:y_tl + size_window, x_tl:x_tl + size_window]
                win_hog = hog_features[ypos:ypos + blocks_per_window, xpos:xpos + blocks_per_window].ravel()
                features = f.get_feature_vector(win_img, subsampled_hog_features=win_hog)
                prediction = self.classifier.predict(features)
                if prediction == 1:
                    x_topleft = scale * x_tl
                    y_topleft = scale * y_tl
                    window_size = scale * size_window
                    box = ((int(x_topleft + x_start), int(y_topleft + y_start)),
                           (int(x_topleft + x_start + window_size), int(y_topleft + y_start + window_size)))
                    rects.append(box)
                    if visualize:
                        cv2.rectangle(visualize_img, box[0], box[1], (255, 0, 0), 3)

        if visualize:
            return rects, visualize_img
        else:
            return rects


if __name__ == '__main__':
    test_img = cv2.imread('test_images/test3.jpg')

    # clf = classifier.train_classifier(*(classifier.read_samples()))
    clf = classifier.Classifier.restore('model')
    pipeline = Pipeline(clf)
    boxes, img = pipeline.search_for_matches(test_img, region_of_interest=((0, 400), (1280, 656)), scale=1.5,
                                             color_space='YCrCb',
                                             visualize=True)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
