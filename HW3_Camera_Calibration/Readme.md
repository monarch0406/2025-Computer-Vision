# **Homework 3 - Camera Calibration**

## 作業目標
本次作業的核心任務為：

1. 根據 3D 世界座標 與 2D 影像座標，計算 Projection Matrix（投影矩陣）  
2. 分解出相機的參數：Intrinsic Matrix（K）、Rotation Matrix（R）、Translation Vector（t）  
3. 驗證投影效果，並計算誤差（RMSE）  
4. 顯示相機姿態（Position & Orientation）

---

## 資料檔案說明

|檔案名稱|內容|用途|
|--------|----|----|
|Point3D.txt|36 個 3D 世界座標點|真實世界定位|
|image1.npy|影像1 的 2D 座標點|圖片上座標|
|image2.npy|影像2 的 2D 座標點|圖片上座標|
|clicker.py|手動標點工具|輔助產生 2D 座標|
|visualize.py|3D 視覺化工具|繪製相機與物件位置|

---

## 作業內容與流程

### 1. 計算投影矩陣（Projection Matrix, P）

#### 目的
- 求得一個 3x4 的矩陣 P
- 能將 3D 世界座標轉換為 2D 影像座標

#### 原理
透過最小平方法（Least Squares）解 Ax = B  
- A 來自 3D 座標  
- B 來自 2D 座標  

#### 公式
3D 世界座標 (X) → 乘上 P → 變成 2D 影像座標 (x)

---

### 2. 分解投影矩陣 → 得到 K、R、t

#### 目的
將投影矩陣 P 分解為：
- K：Intrinsic Matrix（內部參數）  
- R：Rotation Matrix（旋轉矩陣）  
- t：Translation Vector（平移向量）

#### 原理
針對 P 的前 3x3 部分進行 QR 分解 或 Gram-Schmidt 方法，求得 K 與 R，再計算 t。

---

### 3. 驗證結果（Re-Projection & RMSE 計算）

#### 目的
透過計算得到的 K、R、t，將 3D 點重新投影到 2D，並驗證其準確度。計算 RMSE（Root Mean Square Error）作為誤差評估指標。

#### 原理
3D 座標 → 經 K、R、t 轉換 → 應與原始 2D 座標位置接近  
RMSE 值越小，代表結果越準確。

---

### 4. 繪製相機姿態（Camera Pose）

#### 目的
將相機在 3D 空間的：
- 位置
- 方向
進行視覺化展示，並計算兩台相機間的夾角。

---

### 5. 自拍影像處理（擴充應用）

#### 目的
使用自己拍攝的影像（需貼上棋盤圖）進行：
- 手動標記 2D 座標
- 重複執行步驟 1 ~ 4
- 驗證結果

---

## 小結 — 流程圖
```
3D點 + 2D點
     ↓
計算 Projection Matrix (P)
     ↓
分解 P → 得到 K, R, t
     ↓
Re-Projection 驗證 (RMSE)
     ↓
視覺化相機位置 & 姿態
```

---

## 執行環境
- Python 3.x
- NumPy
- OpenCV
- SciPy
- Matplotlib

---

## 補充備註
1. 標記的 2D 點座標順序需與 3D 點對應 → 由左至右、由上至下。  
2. 自動偵測角點可使用 Harris Corner 或其他影像處理方法。  
3. 處理自己拍攝的影像時，流程與原始資料相同，需重新執行步驟 1 ~ 4。

---

## 程式功能對應作業項目 (A ~ F)

|作業項目|程式內容|功能說明|
|--------|--------|--------|
|A. 計算 Projection Matrix|`Projection_Matrix(point2D, point3D)`|給定 3D 與 2D 對應點，使用 least-square 方法計算投影矩陣 P|
|B. 分解 Projection Matrix|`KRt(P)`|將投影矩陣 P 分解為 Intrinsic Matrix (K)、Rotation Matrix (R)、Translation Vector (t)，使用 QR 分解法實作|
|C. Re-Projection 驗證與計算 RMSE|`Verify(P, point3D)`|利用已計算出的 P 將 3D 座標重新投影至 2D 平面，並計算投影誤差 (RMSE)|
|D. Camera Pose 繪製與夾角計算|`visualize.py`|根據求得的 R 與 t，繪製相機位置與姿態，並計算兩台相機之間的夾角|
|E. 自拍影像處理流程|自行重複 A ~ D 步驟|針對自行拍攝的兩張影像，標記 2D 點後，依序執行投影矩陣計算、分解、驗證與繪製|
|F. 自動標記 2D 點|未實作|可額外實作 Harris Corner Detection 或 Hough Transform，自動偵測影像中的角點位置|

---