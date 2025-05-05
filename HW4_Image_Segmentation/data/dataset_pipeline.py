# data/dataset_pipeline.py
import os
import tensorflow as tf

BASE_DIR = "/home/dslab/Monarch/Computer_Vision_HW4/data/train"
IMG_DIR  = os.path.join(BASE_DIR, "input")
GT_DIR   = os.path.join(BASE_DIR, "GT")

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 8
IMG_SIZE = (256, 256)  # 可視需求調整

def parse_example(fname):
    # fname: filename tensor, e.g. "0001.png"
    # 讀 input
    img_path = tf.strings.join([IMG_DIR, fname], separator=os.sep)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

     # 讀 GT：把 "0001.png" → "0001_pix.png"
    mask_fname = tf.strings.regex_replace(
        fname, r"\.png$", "_pix.png"
    )
    mask_path = tf.strings.join([GT_DIR, mask_fname], separator=os.sep)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask, tf.float32)    # ← 加這行，確保後面都是 float32
    # 把 RGB 色標映射到類別編號，例如將 (60,180,90)->0, (110,40,40)->1, ...
    mask = rgb_to_class_indices(mask)
    # one-hot
    mask = tf.one_hot(mask, depth=5)

    return img, mask

def rgb_to_class_indices(mask):
    # 定義顏色對應表
    colors = tf.constant([
        [ 60.,180., 90.],  # 0: Cushion
        [110., 40., 40.],  # 1: Armrest
        [ 50., 10., 70.],  # 2: Leg
        [180.,200., 60.],  # 3: Seat Base
        [100.,100.,100.],  # 4: Backrest
    ])
    # 計算每個像素跟顏色最接近的 index
    mask_flat = tf.reshape(mask, [-1,3])
    dists = tf.reduce_sum((tf.expand_dims(mask_flat,1) - colors)**2, axis=-1)
    class_ids = tf.argmin(dists, axis=1)
    return tf.reshape(class_ids, tf.shape(mask)[:2])

def get_dataset():
    # 列出所有檔名並取交集
    files = tf.io.gfile.listdir(IMG_DIR)
    # 如果有 _pix 差異，先處理檔名對應再過濾
    ds = tf.data.Dataset.from_tensor_slices(files)
    # 解析＋增強
    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(100).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def augment(image, mask):
    # 隨機水平翻轉
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    # 其他增強：亮度、對比度……照需求加
    return image, mask

if __name__ == "__main__":
    ds = get_dataset()
    for imgs, masks in ds.take(1):
        print("Batch images:", imgs.shape, "Batch masks:", masks.shape)
