import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy import ndimage
from scipy.ndimage import convolve

def gaussian_smooth(size, sigma=1):
    """
    Generate a Gaussian kernel for smoothing.
    
    Parameters:
      size  : int, the size of the kernel (must be odd)
      sigma : float, the standard deviation of the Gaussian distribution
      
    Returns:
      kernel: 2D numpy array representing the normalized Gaussian kernel
    """
    # Create a range centered at zero
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    
    # Calculate the 2D Gaussian function
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize the kernel
    
    return kernel

def sobel_edge_detection(im):
    """
    Perform Sobel edge detection on an image.
    
    Parameters:
      im : 2D numpy array (grayscale image) after smoothing
      
    Returns:
      gradient_magnitude : numpy array with gradient magnitudes
      gradient_direction : numpy array with gradient directions (in radians)
    """
    # Define Sobel kernels for horizontal and vertical gradients
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])
    
    # Convolve the image with the Sobel kernels
    gradient_x = ndimage.convolve(im, Kx)
    gradient_y = ndimage.convolve(im, Ky)
    
    # Compute the gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    return (gradient_magnitude, gradient_direction)

def structure_tensor(gradient_magnitude, gradient_direction, k):
    """
    Compute the Harris corner (structure tensor) response.
    
    Parameters:
      gradient_magnitude : numpy array of gradient magnitudes
      gradient_direction : numpy array of gradient directions (in radians)
      k                  : float, Harris detector free parameter (typically 0.04 to 0.06)
      
    Returns:
      R : numpy array, the Harris corner response for each pixel
    """
    # Recover the x and y gradients from the magnitude and direction
    Ix = gradient_magnitude * np.cos(gradient_direction)
    Iy = gradient_magnitude * np.sin(gradient_direction)
    
    # Compute products of derivatives
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy
    
    # Use a simple 3x3 window (you could adjust this window size if needed)
    window = np.ones((3, 3))
    Mxx = ndimage.convolve(Ixx, window)
    Myy = ndimage.convolve(Iyy, window)
    Mxy = ndimage.convolve(Ixy, window)
    
    # Compute the determinant and trace of the second moment matrix
    det_M = Mxx * Myy - Mxy**2
    trace_M = Mxx + Myy
    
    # Compute the Harris response (cornerness measure)
    R = det_M - k * (trace_M**2)
    
    return R

def NMS(harrisim, window_size=30, threshold=0.1):
    """
    Perform Non-Maximum Suppression (NMS) on the Harris response image.
    
    Parameters:
      harrisim    : numpy array, Harris corner response image
      window_size : int, the size of the window to consider for local maxima
      threshold   : float, relative threshold (fraction of the maximum response)
      
    Returns:
      filtered_coords: numpy array of coordinates [row, col] that are considered corners
    """
    # Determine the threshold value based on the maximum Harris response
    thresh_val = threshold * harrisim.max()
    
    # Find the local maximum in the neighborhood defined by window_size
    local_max = ndimage.maximum_filter(harrisim, size=window_size)
    
    # Identify points that are both local maxima and above the threshold
    corner_response = (harrisim == local_max) & (harrisim > thresh_val)
    filtered_coords = np.argwhere(corner_response)
    
    return filtered_coords

def plot_harris_points(image, filtered_coords):
    plt.figure(figsize=(20,10))
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '+')
    plt.axis('off')
    plt.show()
    
def rotate(image, angle, center=None, scale=1.0):
    """
    Rotate the given image by a specific angle.
    
    Parameters:
      image  : input image (numpy array)
      angle  : rotation angle in degrees
      center : tuple (optional), rotation center; defaults to image center
      scale  : scaling factor (default is 1.0)
      
    Returns:
      rotated: the rotated image
    """
    (h, w) = image.shape[:2]
    
    # If no rotation center is provided, use the center of the image
    if center is None:
        center = (w / 2, h / 2)
    
    # Compute the rotation matrix and perform the affine transformation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    return rotated