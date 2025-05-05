import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from tensorflow.keras.models import load_model

# ─── 設定 ─────────────────────────────────────────────────────────
TEST_DIR  = "/home/dslab/Monarch/Computer_Vision_HW4/data/test/input"
GT_DIR    = "/home/dslab/Monarch/Computer_Vision_HW4/data/test/GT"
PRED_DIR  = "/home/dslab/Monarch/Computer_Vision_HW4/predict"
IMG_SIZE      = (256, 256)
NUM_CLASSES   = 5

os.makedirs(PRED_DIR, exist_ok=True)

# ─── 調色盤：class→RGB ─────────────────────────────────────────────
PALETTE = np.array([
    [ 60, 180,  90],   # 0: Cushion
    [110,  40,  40],   # 1: Armrest
    [ 50,  10,  70],   # 2: Leg
    [180, 200,  60],   # 3: Seat Base
    [100, 100, 100],   # 4: Backrest
], dtype=np.uint8)

# ─── RGB mask→class id ──────────────────────────────────────────────
def rgb_to_class_indices(mask):
    colors = tf.constant(PALETTE, dtype=tf.float32)
    m = tf.cast(mask, tf.float32)
    flat = tf.reshape(m, [-1,3])  # (H*W,3)
    dists = tf.reduce_sum((tf.expand_dims(flat,1)-colors)**2, axis=-1)  # (H*W,5)
    cls = tf.argmin(dists, axis=1)
    return tf.reshape(cls, tf.shape(mask)[:2])

# ─── 读取一对测试图 & GT ───────────────────────────────────────────
def load_pair_test(fname):
    # 1) Input
    inp = tf.io.decode_png(
        tf.io.read_file(os.path.join(TEST_DIR,fname)), channels=3
    )
    inp = tf.image.resize(inp, IMG_SIZE)
    inp_np = (inp.numpy().astype(np.uint8))  # for display & PSNR
    
    # 2) GT（带 '_pix' 后缀）
    base, ext = os.path.splitext(fname)
    gt = tf.io.decode_png(
        tf.io.read_file(os.path.join(GT_DIR, base + "_pix" + ext)), channels=3
    )
    gt = tf.image.resize(gt, IMG_SIZE, method="nearest")
    gt_np = gt.numpy().astype(np.uint8)
    gt_cls = rgb_to_class_indices(gt).numpy()
    return inp_np, gt_np, gt_cls

# ─── load model ─────────────────────────────────────────────────────
model = load_model("best.keras")

records = []
for fname in sorted(os.listdir(TEST_DIR)):
    inp_np, gt_np, gt_cls = load_pair_test(fname)
    # predict
    pred = model.predict(inp_np[None,...]/255.0)[0]  # normalize for model
    pred_cls = np.argmax(pred, axis=-1)
    pred_color = PALETTE[pred_cls]  # (H,W,3)

    # build GT color mask
    gt_color = PALETTE[gt_cls]

    # 1) 并排画图
    fig, axes = plt.subplots(1,3, figsize=(9,3))
    axes[0].imshow(inp_np);      axes[0].set_title("Input");  axes[0].axis("off")
    axes[1].imshow(pred_color);  axes[1].set_title("Output"); axes[1].axis("off")
    axes[2].imshow(gt_color);    axes[2].set_title("GT");     axes[2].axis("off")
    save_path = os.path.join(PRED_DIR, fname)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # 2) PSNR & IoU
    psnr = compute_psnr(gt_np, inp_np, data_range=255)
    intersection = np.sum(pred_cls == gt_cls)
    union = pred_cls.size
    iou = intersection / union

    records.append({
        "filename": fname,
        "PSNR": round(float(psnr), 2),
        "IoU":  round(float(iou), 3),
        "out_path": save_path
    })

# ─── 打印 & 存表格 ─────────────────────────────────────────────────
df = pd.DataFrame(records)
print(df.to_markdown(index=False))
df.to_csv(os.path.join(PRED_DIR, "test_metrics.csv"), index=False)
