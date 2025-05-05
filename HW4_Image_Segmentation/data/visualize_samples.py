# data/visualize_samples.py
import os
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')   # 切換到非互動式後端，允許用 savefig
import matplotlib.pyplot as plt

# 路徑設定
BASE_DIR = "/home/dslab/Monarch/Computer_Vision_HW4/data/train"
IMG_DIR  = os.path.join(BASE_DIR, "input")
GT_DIR   = os.path.join(BASE_DIR, "GT")

def load_image_pair(fname: str):
    # input 路徑不變
    img = tf.io.read_file(os.path.join(IMG_DIR, fname))
    img = tf.image.decode_png(img, channels=3)

    # GT 檔名要加上 _pix
    name, ext = os.path.splitext(fname)
    mask_name = f"{name}_pix{ext}"   # e.g. "0361_pix.png"
    mask = tf.io.read_file(os.path.join(GT_DIR, mask_name))
    mask = tf.image.decode_png(mask, channels=3)

    return img, mask


def show_samples(n=3):
    """隨機顯示 n 張影像與對應 Mask"""
    fnames = tf.io.gfile.listdir(IMG_DIR)
    fnames = tf.random.shuffle(fnames)[:n]

    plt.figure(figsize=(8, n*4))
    for i, fname in enumerate(fnames):
        img, mask = load_image_pair(fname.numpy().decode())
        # 原圖
        plt.subplot(n, 2, 2*i+1)
        plt.imshow(img.numpy())
        plt.title(f"Input: {fname.numpy().decode()}")
        plt.axis("off")
        # Mask
        plt.subplot(n, 2, 2*i+2)
        plt.imshow(mask.numpy())
        plt.title("Ground Truth Mask")
        plt.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig("samples.png")
    print("Saved visualize results to samples.png")


if __name__ == "__main__":
    show_samples(4)
