# 2025 Computer Vision
老師 : 楊元福 教授

## HW 1 : Image Sensing Pipeline
練習影像處理，從RGB到RAW，再從RAW到RGB

![alt text](HW1_Image_Sensing_Pipeline/images/image(goal).png)

## HW 2 : Harris Corner Detection

實作哈里斯角點偵測（Harris Corner Detection）及相關影像處理技術，
包含高斯平滑、Sobel 邊緣檢測、結構張量計算、非極大值抑制（NMS）、影像旋轉與縮放。
哈里斯角點偵測是一種用於偵測影像中特徵點的方法，主要識別局部梯度變化劇烈的區域（如邊角）。
透過計算結構張量（Structure Tensor）並應用哈里斯響應函數（Harris Response），可以有效偵測角點。

### **原始圖片**

<p align="center">
    <img src="HW2_Harris_Corner_Detection/original.jpg" width="300">
</p>

---

### **實作函式**

#### **(1) Gaussian smooth results**
**目的：** 使用高斯平滑來降低影像雜訊，以提高邊緣偵測效果。
- **高斯平滑結果（Kernel size = 5, σ = 5）**
- **高斯平滑結果（Kernel size = 10, σ = 5）**

<p align="center">
    <img src="HW2_Harris_Corner_Detection/results/Gaussian smooth results/gaussian_smooth_of_sigma_and_kernal_size_5.jpg" width="300">
    <img src="HW2_Harris_Corner_Detection/results/Gaussian smooth results/gaussian_smooth_of_sigma_and_kernal_size_10.jpg" width="300">
</p>

#### **(2) Sobel edge detection results**
**目的：** 計算影像梯度，以偵測邊緣。

<table align="center">
    <tr>
        <td align="center"><b>梯度大小（Kernel size = 5）</b></td>
        <td align="center"><b>梯度大小（Kernel size = 10）</b></td>
    </tr>
    <tr>
        <td align="center"><img src="HW2_Harris_Corner_Detection/results/Sobel edge detection results/magnitude_of_gradient_kernel_size_5.jpg" width="300"></td>
        <td align="center"><img src="HW2_Harris_Corner_Detection/results/Sobel edge detection results/magnitude_of_gradient_kernel_size_10.jpg" width="300"></td>
    </tr>
    <tr>
        <td align="center"><b>梯度方向（Kernel size = 5）</b></td>
        <td align="center"><b>梯度方向（Kernel size = 10）</b></td>
    </tr>
    <tr>
        <td align="center"><img src="HW2_Harris_Corner_Detection/results/Sobel edge detection results/direction_of_gradient_kernel_size_5.jpg" width="300"></td>
        <td align="center"><img src="HW2_Harris_Corner_Detection/results/Sobel edge detection results/direction_of_gradient_kernel_size_10.jpg" width="300"></td>
    </tr>
</table>

#### **(3) Structure tensor + NMS results**
**目的：** 計算結構張量並應用非極大值抑制，以突顯關鍵角點。
- **角點偵測（視窗大小 = 3×3）**
- **角點偵測（視窗大小 = 30×30）**

<p align="center">
    <img src="HW2_Harris_Corner_Detection/results/Structure tensor + NMS results/NMS_window_size_3.jpg" width="300">
    <img src="HW2_Harris_Corner_Detection/results/Structure tensor + NMS results/NMS_window_size_30.jpg" width="300">
</p>

#### **(4) Final results of rotating**
**目的：** 測試影像旋轉後對角點偵測的影響。
- **旋轉原始影像 30° 後的最終結果**

<p align="center">
    <img src="HW2_Harris_Corner_Detection/results/Final results of rotating/Rotate_30.jpg" width="300">
</p>

#### **(5) Final results of scaling**
**目的：** 測試影像縮放後對角點偵測的影響。
- **將原始影像縮小為 0.5 倍後的最終結果**

<p align="center">
    <img src="HW2_Harris_Corner_Detection/results/Final results of scaling/Scaling.jpg" width="300">
</p>

## HW 3 : Camera Calibration
