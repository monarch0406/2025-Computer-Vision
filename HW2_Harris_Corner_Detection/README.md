# **Homework 2 - Harris Corner Detection**

## **1. Introduction**
This assignment implements the Harris Corner Detection algorithm with various image processing techniques such as Gaussian smoothing, Sobel edge detection, and Non-Maximum Suppression (NMS). The program is structured into multiple Python files and outputs results in a dedicated folder.

## **2. Required Libraries**
The following libraries are required to execute the program:

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters, convolve
```

**Note:**
- `cv2` should **only** be used for image conversion (`BGR2RGB`, `BGR2GRAY`).
- **Do not use** `cv2` functions like `cv2.Sobel`, `cv2.Laplacian`, or `cv2.cornerHarris`.

## **3. Python Files**
### **Main Execution File**
- `hw2.py`: Runs all the required functions and saves the results.

### **Function File**
- `Harris_Corner_Detection.py`: Contains the functions for Harris Corner Detection and image transformations.

## **4. Running the Program**
Execute the following command to run the program and generate images:

```bash
python hw2.py
```

## **5. Implemented Functions**

### **1. Gaussian Smoothing**
```python
def gaussian_smooth(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    img = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return img
```
- Applies Gaussian filtering using the given kernel size and standard deviation (`sigma`).

### **2. Sobel Edge Detection**
```python
def sobel_edge_detection(im, sigma):
    im_smoothed = filters.gaussian_filter(im, sigma)
    gx = filters.sobel(im_smoothed, axis=1, mode='reflect')
    gy = filters.sobel(im_smoothed, axis=0, mode='reflect')
    gradient_magnitude = np.hypot(gx, gy)
    gradient_direction = np.arctan2(gy, gx)
    return gradient_magnitude, gradient_direction
```
- Computes the gradient magnitude and direction using Sobel operators.

### **3. Structure Tensor Computation**
```python
def structure_tensor(gradient_magnitude, gradient_direction, k, sigma):
    Ixx = filters.gaussian_filter(gradient_magnitude**2, sigma)
    Ixy = filters.gaussian_filter(gradient_magnitude * gradient_direction, sigma)
    Iyy = filters.gaussian_filter(gradient_direction**2, sigma)
    det = (Ixx * Iyy) - (Ixy ** 2)
    trace = Ixx + Iyy
    return det / (trace + 1e-12)
```
- Computes the structure tensor matrix and Harris response.

### **4. Non-Maximum Suppression (NMS)**
```python
def NMS(harrisim, window_size, threshold):
    conner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > conner_threshold) * 1
    coords = np.array(harrisim_t.nonzero()).T
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    index = np.argsort(candidate_values)
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[window_size:-window_size, window_size:-window_size] = 1
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0]-window_size):(coords[i, 0]+window_size),
                              (coords[i, 1]-window_size):(coords[i, 1]+window_size)] = 0
    return filtered_coords
```
- Suppresses weak corners to keep only the strongest ones.

### **5. Image Rotation**
```python
def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated
```
- Rotates the image by a specified angle.

## **6. Output Results**
The `results` folder contains five subdirectories with generated images:

### **1. Gaussian Smooth Results**
- `gaussian_smooth_of_sigma_and_kernal_size_5.jpg`
- `gaussian_smooth_of_sigma_and_kernal_size_10.jpg`

### **2. Sobel Edge Detection Results**
- `magnitude_of_gradient_kernel_size_5.jpg`
- `magnitude_of_gradient_kernel_size_10.jpg`
- `direction_of_gradient_kernel_size_5.jpg`
- `direction_of_gradient_kernel_size_10.jpg`

### **3. Structure Tensor + NMS Results**
- `NMS_window_size_3.jpg`
- `NMS_window_size_30.jpg`

### **4. Final Results of Rotation**
- `Rotate_30.jpg`

### **5. Final Results of Scaling**
- `Scaling.jpg`

## **7. Example Image Results**
Below are sample output images:

### **Gaussian Smooth Results**
![Gaussian Smooth](results/Gaussian%20smooth%20results/gaussian_smooth_of_sigma_and_kernal_size_5.jpg)

### **Sobel Edge Detection - Magnitude**
![Sobel Magnitude](results/Sobel%20edge%20detection%20results/magnitude_of_gradient_kernel_size_5.jpg)

### **Sobel Edge Detection - Direction**
![Sobel Direction](results/Sobel%20edge%20detection%20results/direction_of_gradient_kernel_size_5.jpg)

### **Structure Tensor + NMS Results**
![NMS](results/Structure%20tensor%20+%20NMS%20results/NMS_window_size_3.jpg)

### **Final Rotation (30 Degrees)**
![Rotation](results/Final%20results%20of%20rotating/Rotate_30.jpg)

### **Final Scaling (0.5x)**
![Scaling](results/Final%20results%20of%20scaling/Scaling.jpg)

## **8. Conclusion**
This assignment successfully implements Harris Corner Detection with Gaussian smoothing, Sobel edge detection, structure tensor computation, Non-Maximum Suppression, image rotation, and scaling. The results demonstrate how different transformations impact corner detection and image processing.

