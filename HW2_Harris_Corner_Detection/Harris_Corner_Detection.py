import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.ndimage import filters

def gaussian_smooth(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    img =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return img

import numpy as np
from scipy import ndimage

def sobel_edge_detection(im, sigma):
    im_smoothed = ndimage.gaussian_filter(im, sigma)

    gx = ndimage.sobel(im_smoothed, axis=1, mode='reflect')
    gy = ndimage.sobel(im_smoothed, axis=0, mode='reflect')

    gradient_magnitude = np.hypot(gx, gy)  # 梯度大小
    gradient_direction = np.arctan2(gy, gx)  # 梯度方向

    return gradient_magnitude, gradient_direction



def structure_tensor(gradient_magnitude, gradient_direction, k, sigma):
    Ixx=filters.gaussian_filter(gradient_magnitude*gradient_magnitude,sigma)   
    Ixy=filters.gaussian_filter(gradient_magnitude*gradient_direction,sigma)
    Iyy=filters.gaussian_filter(gradient_direction*gradient_direction,sigma)
    det = (Ixx * Iyy) - (Ixy **2)
    trace = Ixx + Iyy
    return det / (trace + 1e-12)

def NMS(harrisim, window_size, threshold):
    conner_threshold=harrisim.max()*threshold
    harrisim_t=(harrisim>conner_threshold)*1
    coords=np.array(harrisim_t.nonzero()).T
    candidate_values=[harrisim[c[0],c[1]] for c in coords]
    index=np.argsort(candidate_values)
    allowed_locations=np.zeros(harrisim.shape)
    allowed_locations[window_size:-window_size,window_size:-window_size]=1
    filtered_coords=[]
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]]==1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-window_size):(coords[i,0]+window_size),(coords[i,1]-window_size):(coords[i,1]+window_size)]=0
    return filtered_coords
    
def rotate(image,angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated