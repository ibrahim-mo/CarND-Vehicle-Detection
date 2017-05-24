## Project Writeup
### Ibrahim Almohandes
### 5/21/2017

---

# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier (for example, Linear SVM)
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_nocar.png
[image2]: ./output_images/hog_y_chan.png
[image3]: ./output_images/hog_sub.png
[image4]: ./output_images/ex_boxes.png
[image5]: ./output_images/heat_fig.png
[image6]: ./output_images/avg_labels.png
[image7]: ./output_images/avg_boxes.png
[video1]: ./project_video.mp4
[video2]: ./project_video-rbf.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)


#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

First, I extracted a selected set of features from the training images (`vehicles` and `non-vehicles`), then ran a classifier on them, and finally saved the parameters and trained classifier to a pickle file. I wrote a separate file for this step, and called it `extract_features.py`.

The code for extracting the features (of both cars and not-cars) is contained at lines 37 through 100 of the file `extract_features.py`. The function `extract_features()` (defined at lines 37 to 81 of the file) calls support functions which I defined in another file called `supp_funcs.py`. I did this because I will need some of these lower-level functions again in the next steps (for vehicle detection and tracking).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Then, I explored different color spaces and different `skimage.feature.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

The parameters I used for extracting image features are as follows:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

The above parameters are passed by `extract features()` as inputs to the following functions: `convert_color()`, `get_hog_features()`, `bin_spatial()`, and `color_hist()`. These functions are defined at lines 7 through 55 of the file `supp_funcs.py`.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` [I am showing the Y-Channel hog as an example]:


![alt text][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters for different sets of features. Here are some of the important findings:

- For color space, I tried RGB, HSV, and YCrCb, then I found that YCrCb performs better. Color space affects all three feature sets (HOG, spacial bins, and color histograms).

- For HOG channels, I found that selecting 'ALL' channels gives cleaner results than choosing only one channel (such as the Y-channel).

- One important thing - that was also mentioned in the lectures - is that if we train our classifier on PNG images, then try to predict classes (i.e., find cars) in JPEG images, or vice versa, we may get unexpected results. That is because `mpimg.imread()` converts PNG's RGB values into the real range `[0.0, 1.0]`, while for JPEG images it uses the normal (integer) range `[0, 255]`. To resolve this issue, I do the following conversion when I extract features from PNG images (then later when I detect cars from video frames, RGB values will be already in sync with the trained classifier):

```python
image = (image*255).astype(np.uint8)
```

- Last, but not least, was the choice of the classifier and kernel. I will explain this in the following point.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After extracting the concatenated features, I normalized them with the `scikit-learn`'s `StandardScaler()`, which standardizes the combined image features with zero mean and unit variance.

Then, I chose the SVM classifier, but tried three different kernels: `LinearSVC()`, `SVC()` with default `kernel='rbf'`, and `SVC(kernel='poly')` with default `degree=3`. I found that the linear SVC was the fastest of the three, the polynomial SVC was extremely slow (crashed or ran out of memory after several hours), and the RBF SVC was in between but it was still very slow (more than 20 times slower than `LinearSVC()`, although I ran it on a t2.txlarge AWS instance).

I finally chose to train the extracted features with the Linear SVM classifier (`LinearSVC()`), although the RBF-kernel SVM classifier (default `SVC()`) did slightly better on examples images and the test video, but due to speed it was not practical to experiment with it within a reasonable amount of time.

The code for normalizing the image features and training the Linear SVC classifier is at lines 102 through 131 of file `extract_features.py`


### Sliding Window Search


#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To find cars in an image (or a video frame) using the trained SVM classifier, I applied a sliding window approach using the method described in the lectures, which is combined with a HOG subsampling technique to improve performance. The hog features are calculated for the selected hog channels (in my case, hog_channel='ALL'), then the (three) feature vectors (per image) are subsampled into windows that can be scaled to different ratios, one at a time.

For the parameters `orientations`, `pixels_per_cell`, and `cells_per_block`, these are the same as the ones that were used in the feature extraction step (i.e., 9, 8, 2 respectively). In fact, I load these values from the pickle file that was saved in the feature extraction step.

There are a few extra parameters that remain to be chosen. These are related to previously selected values, and third one is the most important:

- A window size of 64 pixels (per square side), which is based on the earlier selection of 8 (cells_per_block) x 8 (pixels_per_cell),

- A window overlap of 1 cell (which is similar to the cell overlap used in the HOG feature extraction)

- A varying window scaling factor. I experimented with multiple values, and found that the following combination of scaling factors works fairly well: `[1.0, 1.5, 2.0]`. For each image (or video frame), the image pipeline loops through all three scales and adds the found boxes to a combined set of detected boxes.

Here's an example of an image after applying the sliding window search (with hog subsampling) using a scale factor of `1.5`.

![alt text][image3]

The code for the sliding window implementation is defined in the function `find_cars()` at lines 26 through 98 in the file `proj5.py`. Similar to what was done earlier, this function calls lower-level functions defined in the file `supp_funcs.py()` (at lines 7 to 55).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video, using multiple scale factors (`[1.0, 1.5, 2.0]`).  From the positive detections, I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are five frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all five frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

This image pipeline for creating video frames is defined by the function `img_pipeline()` at lines 110 through 149 of the file `proj5.py`.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I tried multiple techniques during feature extraction like selecting a good color scheme such as YCrCb, and combining multiple features like HOG channels, color histograms, and spacial binning. I normalized the features before training, then used an SVM classifier. 

I did several steps experimenting with the linear SVM classifier. The good thing is that it was much faster than the other SVC kernels, hence I had the chance to run multiple rounds with different combination of parameters, color schemes, and hog channels, as well as window scaling factors. Another SVC() classifier, such as RBF-kernel SVC, could do better but it was extremely slow, and hence not practical to train and experiement with it in a reasonable amount of time.

I used a sliding window technique along with an efficient subsampling method to find cars in each video frame, and extract features only once per image, then subsample the features for overlapping windows using different scales for a better search. Then I used a labeling method to find groups of pixels (car candidate locations) from all the bounding boxes, and did filtering with a small threshold (to reduce false positives). Finally, I kept track of the last five video frames, and took their avergae to draw the boxes on each frame. That simple technique helped provide a somewhat better stabilization of the drawn boxes, as well as reduction in the number of false positives.

I noticed that my pipeline worked fairly well with cars that are closer (as they appear bigger), but it didn't do well finding cars that are far at the horizon (or appear small). Also, there were a (very) few false positives. Ideally, it wound be nicer to discover all visible cars with no false positives (like humans do!).

I already tried a better SVM classifier which is `SVC()` with default RBF kernel (which is much slower than `LinearSVC()`), but at that time I didn't use the frame avergaing method and hence I was asked to resubmit the project with it. Unfotunately, I didn't have sufficient time to create another output video with this method. You can actually check [my RBF-based video here](./output_video-rbf.mp4). It discovered more cars and had no false postivies (without using frame. Also, a deep neural network could be more robust than a linear SVM which is something I could try in the future.

Another nice thing to try in the future is to mix the output of this video with the lane lines produced by the previous project!

Thanks!
