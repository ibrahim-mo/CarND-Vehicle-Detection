import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from scipy.ndimage.measurements import label
from collections import deque

from supp_funcs import *

# load the extracted features from the pickle file
dist_pickle = pickle.load( open("svc_pickle-lin.p", "rb") )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
color_space = dist_pickle["color_space"]
hog_channel = dist_pickle["hog_channel"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)

    # for PNG images read with mpimg.imread, convert range back to [0, 255]
    # if mpimg_png:
    #     img = (img*255).astype(np.uint8)

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    bboxes = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1, hog_feat2, hog_feat3 = [], [], []
            if hog_channel==0 or hog_channel=='ALL':
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel==1 or hog_channel=='ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel==2 or hog_channel=='ALL':
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                xbox, ybox = (xbox_left, ytop_draw+ystart), (xbox_left+win_draw,ytop_draw+win_draw+ystart)
                cv2.rectangle(draw_img,xbox,ybox,(0,0,255),4)
                bboxes.append((xbox,ybox))
                
    return draw_img, bboxes
    
ystart = 400
ystop = 656
# Try different scales
scales = [1., 1.5, 2.]
# heat threshold
heat_thr = 3

# For storing history of the last N (=5) heatmaps
history = deque(maxlen = 5)

def img_pipeline(img):
    if not img.any():
        return None

    box_list = []

    for scale in scales:
        out_img, bboxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        if bboxes:
            box_list.append(bboxes)

    if box_list:
        box_list = np.concatenate(box_list)

    # heat canvas to draw on
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, box_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, heat_thr)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Take average of the last 5 heatmaps
    history.append(heatmap)
    avg_heatmap = np.mean(history, axis=0)

    # Find final boxes from heatmap using label function
    # labels = label(heatmap)
    labels = label(avg_heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    #save frames into files
    # t = time.time()
    # mpimg.imsave('tmp/frame'+str(t)+'.png', draw_img)

    return draw_img

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

test=0
if test:
    video_output = "outtest_video.mp4"
    clip1 = VideoFileClip("test_video.mp4")
else:
    video_output = "output_video.mp4"
    clip1 = VideoFileClip("project_video.mp4")
out_clip = clip1.fl_image(img_pipeline) #NOTE: this function expects color images!!
out_clip.write_videofile(video_output, audio=False)
