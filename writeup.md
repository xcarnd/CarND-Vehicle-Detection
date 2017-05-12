**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/dataset_explore.png
[image2]: ./output_images/hog_features.png
[image3]: ./output_images/sliding_window_search.png
[image4]: ./output_images/heatmap.png
[image5]: ./output_images/labeled_heatmap.png
[image6]: ./output_images/output.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 9 through 29 of the file called `features.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Dataset Exploration][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG features][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters.

I first played with the color space parameter. I've tried RGB, HSL, LUV, YCrCb. Among them LUV/YCrCb gave me the highest test accuracy when training the classifier. When tested with the test images, YCrCb yields fewer false positive. So I finally used YCrCb.

After the color space, I tuned the parameters for HOG features extraction. I found using all channels can gave me the highest testing accuracy. Different orientations, pixels_per_cell, cells_per_block did not result in big differences in accuracy. Higher pixels_per_cell/cells_per_block usually get a little penalty in accuracy, but also result in less output HOG features and so extraction is faster.

The final choice of parameters of mine are the same as the ones used in video lectures, that is:

| Parameter         | Value    |
|:-----------------:|:--------:|
| Color space       | YCrCb    |
| orientation       | 9        |
| pixels_per_cell   | 8        |
| cells_per_block   | 2        |
| channel           | -1 (ALL) |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the default parameter settings. Codes can be found in `classifier.py`. The whole file is all about training a classifier.

After all samples are read in, I converted each sample to YCrCb color space, then extracted HOG features, color histogram features and spatial binning features from each sample. Codes are contained in `classifier.py` lines 17 through 21 and `features.py` lines 50 through 76.

Codes for color histogram features can be found in lines 32 through 38 in `features.py`. I applied `numpy.histogram` with `nbins=32` to all the channels of the image individually, then concatenated into a 1D feature vector.

Codes for spatial binning features can be found in lines 41 through 47 in `features.py`. Each channel of the image is downsampled to a 16 x 16 image, then concatenated into a 1D feature vector.

With the feature vectors and labels prepared, I normalized them by using `StandardScaler`. Codes are contained in lines 103 through 105 in `classifier.py`.

Then the data set is split into a training set and a validation set for training and validating training result for a linear SVC. The trained SVC and scaler are then pickled for future use.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I performed sliding window search with the following step:

    i. Limit the searching area within a region of interests. In practice, I excluded the area where cars would definitely not showing (i.e., regions above the horizons).
    
    ii. If scaling is needed, scales the searching area accordingly.
    
    iii. Perform sliding window search with the scaled search area image.

Codes can be found in lines 60 through 155 in `vehicle_detection.py`.

The scales I used for searching are 1.0 and 1.5. 1.5 is the main scale giving out most of the detections, while scale at 1.0 can be helpful for detecting small car images in some cases.

I used 75% overlapping of windows. For higher overlap I can get more results for a single target, which may be helpful when I need to kick off outlier, but searching is slower. On the other hand, lower overlap gives out less results but also make distinguishing outlier harder (by checking with heatmap).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some example images:

![Sliding Window Search][image3]

To improve the performance of my classifier, I used HOG sub-sampling to reduce the time spent on HOG features calculation. I also limited the searching area for smaller scales by the fact that smaller scales are used for detecting smaller car images, which means the cars are far away from the camera and would finally appear on the upper part of the searching region.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap. Heatmaps for the last 5 video frames are kept and then combined into an integrated heatmap. Thresholding is performed on the integrated heatmap to throw away false positives.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the integrated heatmap. Each blob is assumed to be corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

Codes for the process are contained in lines 22 through 58 in `vehicle_detection.py`

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 5 frames and their corresponding heatmaps:

![Individual Heatmaps and Bounding boxes][image4]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 5 frames:
![Labeled Heatmap][image5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![Output][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major issue of my implementation is the detection is pretty time consuming. It spend a lot of time in extracting HOG features.

Despite of the time it takes, the main problem of my implementation is it may not be working well in different surroundings. The training set I used contains some samples drawing from the video stream of the project, so I can feed them to the classifier and telling it "this is not a vehicle". But if the surroundings is changed and no similar samples are already contained in the training set, the classifier may fail in identifying "this is not a vehicle", leading to more false positive. To solve this, I think, data augmentation is a primary mean to make the detection more robust.

