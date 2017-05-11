# -*- encoding: utf-8 -*-

import cv2
import utils
import features as f
import matplotlib.pyplot as plt
import numpy as np
import pickle
import settings as s
from scipy.ndimage.measurements import label


class Pipeline(object):
    searching_scales = [1.0, 1.5, 2.0]

    def __init__(self, classifier, scaler):
        self.classifier = classifier
        self.scaler = scaler
        self._last_heatmaps = []

    def search_cars(self, image, region_of_interest=None, sequence=True):
        heatmap = np.zeros(image.shape[:2], dtype=np.float)
        for scale in self.searching_scales:
            boxes = self.search_for_matches(image, region_of_interest=region_of_interest, scale=scale)
            hm = build_heatmap((image.shape[1], image.shape[0]), boxes)
            heatmap += hm

        if sequence:
            self._last_heatmaps.append(heatmap)
            if len(self._last_heatmaps) > 5:
                del self._last_heatmaps[0]
            accumulated_heatmap = np.sum(self._last_heatmaps, axis=0, keepdims=False)
            accumulated_heatmap = apply_heatmap_threshold(accumulated_heatmap, 20)

            accumulated_heatmap = np.clip(accumulated_heatmap, a_min=0, a_max=255)
        else:
            accumulated_heatmap = heatmap

        max_val = np.max(accumulated_heatmap)
        scaled = accumulated_heatmap * 255 / max_val
        return np.stack((scaled, scaled, scaled), axis=-1).astype(np.uint8)
        # bboxes = label_heatmap_and_get_bounding_box(heatmap)
        # img = get_image_with_boxes(test_img, bboxes)

    def search_for_matches(self, image, region_of_interest=None, scale=1.0, visualize=False):
        """Apply sliding window search on the given image.
        
        :param image: the region which search is imposed on.
         
        :param region_of_interest: region in which the search is limited in. If unspecified (None), defaults to the  
            full region of the image. Specified in the format:
            `((top-left-x, top-left-y), (bottom-right-x, bottom-right-y))`
        
        :param scale: Searching window scales.

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
        search_region = utils.convert_color_space(search_region, s.color_space)
        # print("Shape of search region: ", search_region.shape)

        # scaling the input if necessary
        if scale != 1:
            search_region = cv2.resize(search_region,
                                       (int(search_region.shape[1] / scale), int(search_region.shape[0] / scale)))
            # print("Scaled shape of search region: ", search_region.shape)

        # cars looked smaller and closer to the horizon. so I can limit the searching area
        # for smaller scale (which is used for searching for "small" car) to the upper part
        # of the search region
        crop = min((0.5 * scale, 1))
        search_region = search_region[:int(crop*search_region.shape[0]), :]

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
                win_hog = hog_features[:, ypos:ypos + blocks_per_window, xpos:xpos + blocks_per_window].ravel()
                features = f.get_feature_vector(win_img, subsampled_hog_features=win_hog)

                scaled_features = self.scaler.transform(features.reshape(1, -1))
                prediction = self.classifier.predict(scaled_features)
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


def build_heatmap(size, boxes):
    heatmap = np.zeros((size[1], size[0]))
    for ((tl_x, tl_y), (br_x, br_y)) in boxes:
        heatmap[tl_y:br_y, tl_x:br_x] += 1
    return heatmap


def apply_heatmap_threshold(heatmap, threshold=0):
    heatmap[heatmap < threshold] = 0
    return heatmap


def label_heatmap_and_get_bounding_box(heatmap):
    bboxes = []
    labels = label(heatmap)
    num = labels[1]
    for n in range(1, num + 1):
        nonzero = (labels[0] == n).nonzero()
        nzx, nzy = np.array(nonzero[1]), np.array(nonzero[0])
        x1, y1 = np.min(nzx), np.min(nzy)
        x2, y2 = np.max(nzx), np.max(nzy)
        bboxes.append(((x1, y1), (x2, y2)))
    return bboxes


def get_image_with_boxes(image, boxes, color=(255, 0, 0), thickness=3):
    img = np.copy(image)
    for (p1, p2) in boxes:
        cv2.rectangle(img, p1, p2, color=color, thickness=thickness)
    return img


def constructor_pipeline_from_classifier(pickle_file_name):
    with open(pickle_file_name, 'rb') as clf_data_file:
        clf_data = pickle.load(clf_data_file)
        clf = clf_data['classifier']
        scaler = clf_data['scaler']
        clf_data_file.close()
    return Pipeline(clf, scaler)


if __name__ == '__main__':
    pipeline = constructor_pipeline_from_classifier("clf.p")
    # # import os
    # # frames = os.listdir("debug")
    # # for frame in frames:
    # #     test_img = cv2.imread('debug/{}'.format(frame))
    # #     result = pipeline.search_cars(test_img, region_of_interest=((0, 400), (1280, 656)))
    # #     cv2.imwrite('debug2/{}'.format(frame), result)
    # test_img = cv2.imread('debug/frame_020.jpg')
    # # result = pipeline.search_cars(test_img, region_of_interest=((0, 400), (1280, 656)), sequence=False)
    # boxes, result = pipeline.search_for_matches(test_img, region_of_interest=((0, 400), (1280, 656)), visualize=True)
    # plt.imshow(result)
    # plt.show()

    # import os
    # frames = os.listdir("debug")
    # for frame in frames:
    #     test_img = cv2.imread('debug/{}'.format(frame))
    #     boxes, result = pipeline.search_for_matches(test_img, region_of_interest=((0, 400), (1280, 656)),
    #                                                 visualize=True, scale=2.0)
    #     plt.imshow(result)
    #     plt.show()
