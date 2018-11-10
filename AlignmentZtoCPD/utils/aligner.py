import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import skimage.transform as tf
import scipy.signal as signal
import re
import sys

class Aligner(object):
    def __init__(self):
        self.max_features = 50000
        self.good_features = 0.1
        self.upsampling = 8

    # returns transform and warped image
    def alignImages(self, source_orig, target_orig):
        (height, width) = source_orig.shape

        # Scale to uint8 in preparation for ORB
        target = np.interp(target_orig, (target_orig.min(), target_orig.max()), (0, (2**8 - 1)))
        target = target.astype(np.uint8)

        source = np.interp(source_orig, (source_orig.min(), source_orig.max()), (0, (2**8 - 1)))
        source = source.astype(np.uint8)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(self.max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(source, None)
        keypoints2, descriptors2 = orb.detectAndCompute(target, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Keep good_features*100% of features
        numGoodMatches = int(len(matches) * self.good_features)
        matches = matches[:numGoodMatches]

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        trans, mask = cv2.estimateAffine2D(points1, points2, cv2.RANSAC)
        
        # Remove rotation and scaling
        # trans[0:2,0:2] = np.identity(2)

        print("Transform identified:")
        print(trans)
        # Use homography
        height, width  = target.shape
        source_registered = cv2.warpAffine(source_orig, trans, (width, height))

        return source_registered, trans

    # return largest rectangle that fits within the image space
    def autoCropper(self, img):
        # Scale to uint8 in preparation for findContours
        img = np.interp(img, (img.min(), img.max()), (0, (2**8 - 1)))
        img = img.astype(np.uint8)

        _,thresh = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        hull = cv2.convexHull(cnt)

        # fit what is hopefully a quadrilateral to the image
        simplified_cnt = cv2.approxPolyDP(hull,0.001*cv2.arcLength(hull,True),True)[:,0]
        if len(simplified_cnt) != 4:
            raise Exception("Something went wrong during cropping.")
        simplified_cnt[:,0].sort()
        simplified_cnt[:,1].sort()
        # find maximum internal rectangle for crop
        xmin, xmax = simplified_cnt[:,0][1:3]
        ymin, ymax = simplified_cnt[:,1][1:3]
        
        return xmin, xmax, ymin, ymax

    # resize and pad image, maintain aspect ratio
    def autoPadder(self, img, width, height):
        (original_height, original_width) = img.shape
        ratio_x = width/original_width
        ratio_y = height/original_height
        if ratio_x == ratio_y:
            return cv2.resize(img, (width, height), cv2.INTER_LANCZOS4)
        
        #resize by smaller ratio
        elif ratio_x < ratio_y:
            new_width = int(np.rint(original_width*ratio_x))
            new_height = int(np.rint(original_height*ratio_x))
            img_temp = cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)
            
            #Calculate blank space (pad right extra if odd)
            pad_up = int((height-new_height)//2)
            pad_down = int((height-new_height)//2) + (height-new_height)%2
            
            return np.pad(img_temp, [(pad_up,pad_down),(0,0)], 'constant', constant_values=0)
            
        else:        
            new_width = int(np.rint(original_width*ratio_y))
            new_height = int(np.rint(original_height*ratio_y))
            img_temp = cv2.resize(img, (new_width, new_height), cv2.INTER_LANCZOS4)
            
            #Calculate blank space (pad right extra if odd)
            pad_left = int((width-new_width)//2)
            pad_right = int((width-new_width)//2) + (width-new_width)%2
            
            return np.pad(img_temp, [(0,0),(pad_left,pad_right)], 'constant', constant_values=0)
    
    def sharpen(self, img):
        blur = cv2.GaussianBlur(img, (25,25), 20)
        return cv2.addWeighted(blur, -2, img, 3,0)
    
    def upsample(self, img):
        (height, width) = img.shape
        new_dims = (width*self.upsampling, height*self.upsampling)
        return cv2.resize(img, new_dims, cv2.INTER_LANCZOS4)