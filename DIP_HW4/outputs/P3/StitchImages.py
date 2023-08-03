
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import itertools


# In[ ]:


def registration(img1, img2, ratio):
    
    # Initialization
    
    min_match = 1
    
    # Compute SIFT features
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Find matches
    #matcher = cv2.BFMatcher()
    #raw_matches = matcher.knnMatch(des1, des2, k=2)
    flann_params = dict(algorithm=1,
                        trees=5)
    flann = cv2.FlannBasedMatcher(flann_params, {})
    raw_matches = flann.knnMatch(des1, des2, k=2)
    
    # Keeping good matches
    good_points = []
    good_matches=[]
    
    for m1, m2 in raw_matches:
        if m1.distance <  ratio * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])
    
    # Calculate Homography
    if len(good_points) >  min_match:
        image1_kp = np.float32(
            [kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32(
            [kp2[i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,4.0)

    # The more two images have good points, the more they match
    return H, -np.sum(H)/np.sum(status)


# In[ ]:


def stitch(img1,img2,H):
    #H,match = registration(img1,img2)
    
    # Get width and height of input images
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([[0, 0], [0, w1], [h1, w1], [h1, 0]]).reshape(-1, 1, 2)
    img2_dims_temp = np.float32([[0, 0], [0, w2], [h2, w2], [h2, 0]]).reshape(-1, 1, 2)

    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, H)

    # Resulting dimensions
    result_dims = np.concatenate((img1_dims, img2_dims), axis=0)

    # Combining the images
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    # Warp images to get the resulting image
    result = cv2.warpPerspective(img2, transform_array.dot(H),
                                     (x_max - x_min, y_max - y_min))

    result[transform_dist[1]:w1 + transform_dist[1],
    transform_dist[0]:h1 + transform_dist[0]] =+ img1

    return result


# In[ ]:


def stitch_image(images,ratio=0.75):
    
    while(len(images)>1):
        
        # Finding the best match
        comb = list(itertools.combinations(range(len(images)), 2))
        min = np.inf
        for i in range(len(comb)):
            h, match = registration(images[comb[i][1]],images[comb[i][0]],ratio)
            if match<min:
                min=match
                H=h
                ind = i
        # Stitching the two images
        images[comb[ind][0]] = stitch(images[comb[ind][1]], images[comb[ind][0]],H)
        res = images[comb[ind][0]].copy()
        del images[comb[ind][1]]
    
    return res

