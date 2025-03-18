# **Homework 2 Instructions**  

## **1. Required Libraries:**  
The following libraries will be used in this assignment:  
`os`, `cv2`, `numpy`, `matplotlib`, `scipy`  

**Notice:** `cv2` should only be used for image conversion (e.g., `BGR2RGB`, `BGR2GRAY`). It **must not** be used directly for the required code in this assignment (e.g., `cv2.Sobel`, `cv2.Laplacian`, `cv2.cornerHarris`).

## **2. Python Files:**  
- `hw2.py` (Main execution file)  
- `Harris_Corner_Detection.py` (Function file)  

## **3. Running the Program:**  
To execute the program and generate images, enter the following command:  
```bash
python hw2.py
```  

## **4. Functions Included:**  
- `gaussian_smooth`  
- `sobel_edge_detection`  
- `structure_tensor`  
- `NMS` (Non-Maximum Suppression)  
- `rotate`  

## **5. Contents of the `results` Folder:**  
The `results` directory contains five subfolders with the following outputs:  

### **(1) `Gaussian smooth results`**  
Contains **2 images**:  
- **Gaussian smooth results:** \( \sigma = 5 \), kernel size = 5  
- **Gaussian smooth results:** \( \sigma = 5 \), kernel size = 10  

### **(2) `Sobel edge detection results`**  
Contains **4 images**:  
- **Magnitude of gradient** (Gaussian kernel size = 5 and 10) → **2 images**  
- **Direction of gradient** (Gaussian kernel size = 5 and 10) → **2 images**  

### **(3) `Structure tensor + NMS results`**  
Contains **2 images**:  
- Window size = **3×3**  
- Window size = **30×30**  

### **(4) `Final results of rotating`**  
Contains **1 image**:  
- **Final result after rotating** the original image by **30°**  

### **(5) `Final results of scaling`**  
Contains **1 image**:  
- **Final result after scaling** the original image to **0.5×**  