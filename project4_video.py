# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 08:54:26 2017

@author: Admin
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

# CALIBRATE 
def calibrate():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('../camera_cal/calibration*.jpg') # glob.glob('calibration_wide/GO*.jpg')
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            
    
    
    
    # Take any image
    img = mpimg.imread('../camera_cal/calibration6.jpg')
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "../camera_cal/wide_dist_pickle.p", "wb" ) )

# UNDISTORT
def undistort_image(img):
    # Read in the saved camera matrix and distortion coefficients
    # These are the arrays you calculated using cv2.calibrateCamera()
    dist_pickle = pickle.load( open(  "../camera_cal/wide_dist_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
#230
def thr_pipeline(img, h_thresh = (0, 230), s_thresh=(170, 255), sx_thresh=(20, 100), sy_thresh=(20, 100), grad_thresh=(np.pi/4, np.pi/2)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1
    
     # Sobel y
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # Threshold x gradient
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1
    
    # Magnitude threshold
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    sgradbinary =  np.zeros_like(absgraddir)
    sgradbinary[(absgraddir >= grad_thresh[0]) & (absgraddir <= grad_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    combined_binary = np.zeros_like(sxbinary)
    # Combine the two binary thresholds
    combined_binary[(s_binary == 1) | ((h_binary == 1) & (sxbinary == 1) & (sybinary == 1) & (sgradbinary == 1))] = 1
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return combined_binary



def warp_image(img):
    src = np.float32([[200, 720], [590, 450], [685, 450], [1100, 720]]);
    dst = np.float32([[340, 720], [340, 0], [920, 0], [920, 720]]);
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, Minv

def find_lines(binary_warped):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
        
    return left_fit, right_fit

def find_lines_video(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    try:
       left_fit = np.polyfit(lefty, leftx, 2)
       right_fit = np.polyfit(righty, rightx, 2)
    except:
       left_fit = [0,0,0]
       right_fit = [0, 0, 0] 
    return left_fit, right_fit 

# finding the radius
def find_radius(ploty, left_fit, right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    img_center = 1280/2 # center of the image
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    y_eval = np.max(ploty) #binary_warped.shape[0] 
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    curv = (left_curverad + right_curverad)/2 # curvature
    # the deviation of the midpoint of the lane from the center of the image is the positon
    pos = (img_center - np.mean( right_fitx + left_fitx)/2)*xm_per_pix # vehicle position 
    return curv, pos

#sanity check for lines
def sanity_check(ploty, left_fit, right_fit):
    
    CURVATUR_DIFF = 2
    MAX_POS_DIFF = 8
    MIN_POS_DIFF = 3
    PAR_DIFF = 2
    RAD_THR_LOW = 1
    RAD_THR_HIGH = 5000
    POS_THR_LOW = 0
    POS_THR_HIGH = 5
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    check = False
    
    rad_check = False
    curv, pos = find_radius(ploty, left_fit, right_fit)
    
    
    if (curv >= RAD_THR_LOW) & (curv <= RAD_THR_HIGH) & (abs(pos) >= POS_THR_LOW) & (abs(pos) <= POS_THR_HIGH):
        rad_check = True
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    if rad_check & (abs(left_fit[0] - right_fit[0]) < CURVATUR_DIFF) & (np.mean(abs(left_fitx - right_fitx))*xm_per_pix < MAX_POS_DIFF) & (np.mean(abs(left_fitx - right_fitx))*xm_per_pix > MIN_POS_DIFF) & (np.std(abs(left_fitx - right_fitx))*xm_per_pix  < PAR_DIFF):
       check = True
    return check    
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit_left = [np.array([False])]
         #polynomial coefficients for the most recent fit
        self.current_fit_right = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        # number of false line detections
        self.n_false_lines = 0
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

NFALSE_LINES_THR = 10 # number of consecutive false lines before windows searching
MAVER = 0.9 # moving average constant 

RAD_THR_LOW = 1
RAD_THR_HIGH = 5000
    
l = Line()

#n = 0    
#CALIBRATE
#calibrate()
   
# Read in an image
#img = mpimg.imread('../test_images/test6.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Image processing pipeline
def process_image(img):
    #UNDISTORT
    undist = undistort_image(img)
    
    #global n
    #n = n + 1
   # if n == 10:
    #   mpimg.imsave("../video_images/" +'image'+ str(n) + '.jpg',img)
       
    #THRESHOLD 
    thr_image = thr_pipeline(undist)
          
    #WARP
    warped, Minv = warp_image(thr_image)
    
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
     
    # FIND LINES aproximate window search
    if l.detected:
       left_fit, right_fit = find_lines_video(warped, l.current_fit_left, l.current_fit_right)
       #left_fit, right_fit = find_lines(warped)
       if ((np.sum(left_fit) > 0) & (np.sum(right_fit) > 0)) & sanity_check(ploty, left_fit, right_fit):
           # take into account only reliable detections
        
          l.current_fit_left = MAVER*l.current_fit_left + (1-MAVER)*left_fit
          l.current_fit_right = MAVER*l.current_fit_right + (1-MAVER)*right_fit
          l.n_false_lines = 0
       else:
           l.n_false_lines = l.n_false_lines + 1
           if l.n_false_lines > NFALSE_LINES_THR:
              left_fit, right_fit = find_lines(warped)
             
              
              l.current_fit_left = MAVER*l.current_fit_left + (1-MAVER)*left_fit
              l.current_fit_right = MAVER*l.current_fit_right + (1-MAVER)*right_fit
              l.n_false_lines = 0
    else:                                          
    # FIND LINES by windowed searsch
       left_fit, right_fit = find_lines(warped)
       l.current_fit_left = left_fit
       l.current_fit_right = right_fit
       l.detected = True 
    # FIND RADIUS
    curv, pos = find_radius(ploty, l.current_fit_left, l.current_fit_right)
    
    if l.radius_of_curvature == None:
       l.radius_of_curvature = curv
       l.line_base_pos = pos
    else:
         if (curv > RAD_THR_LOW) & (curv < RAD_THR_HIGH):
            l.radius_of_curvature = MAVER*l.radius_of_curvature + (1-MAVER)*curv
            l.line_base_pos = MAVER*l.line_base_pos + (1-MAVER)*pos
       
    # DRAW LINES ON ORGINAL IMAGE
   
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
   
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    
    # put text
    font                   = cv2.FONT_ITALIC
    bottomLeftCornerOfText = (50,650)
    fontScale              = 1.2
    fontColor              = (255,0,0)
    lineType               = 3
    
    if l.line_base_pos < 0:
       text = 'radius ' + str(round(l.radius_of_curvature,2)) + 'm' +'   veh. pos. ' + str(round(abs(l.line_base_pos),2)) + 'm' + ' left of the center'
    else:
        text = 'radius ' + str(round(l.radius_of_curvature,2)) + 'm' +'   veh. pos. ' + str(round(abs(l.line_base_pos),2)) + 'm' + ' right of the center'
        
    cv2.putText(result, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    #Display the image
    return result

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
#from IPython.display import HTML

white_output = '../project_video_output1.mp4'

##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("../project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)