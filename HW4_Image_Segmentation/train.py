import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from data.dataset_pipeline import parse_example, augment, AUTOTUNE, BATCH_SIZE
from models.unet import build_unet

# ─── 參數設定 ─────────────────────────────────────────────────────
EPOCHS      = 20
LR          = 1e-4
IMG_DIR     = "/home/dslab/Monarch/Computer_Vision_HW4/data/train/input"
VAL_DIR     = "results/val"
NUM_CLASSES = 5

# ─── 準備資料清單 & 切分 train/val ─────────────────────────────────
all_files = os.listdir(IMG_DIR)
random.shuffle(all_files)
split_idx   = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files   = all_files[split_idx:]

# ─── Dataset 建構函式 ──────────────────────────────────────────────
def make_dataset(file_list, augment_fn=True):
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    ds = ds.map(parse_example, num_parallel_calls=AUTOTUNE)
    if augment_fn:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(100).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_files, augment_fn=True)
val_ds   = make_dataset(val_files,   augment_fn=False)

# ─── 建立結果目錄 ───────────────────────────────────────────────────
os.makedirs(VAL_DIR, exist_ok=True)

# ─── 調色盤：class 0–4 對應 RGB，用於視覺化 ──────────────────────────
PALETTE = np.array([
    [ 60, 180,  90],  # 0: Cushion
    [110,  40,  40],  # 1: Armrest
    [ 50,  10,  70],  # 2: Leg
    [180, 200,  60],  # 3: Seat Base
    [100, 100, 100],  # 4: Backrest
], dtype=np.uint8)

# ─── 載入模型、編譯 ─────────────────────────────────────────────────
model = build_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss="categorical_crossentropy",
    metrics=[tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)]
)

# ─── Callback：存最佳模型 ───────────────────────────────────────────
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best.keras",
    save_best_only=True,
    monitor="val_loss"
)

# ─── Callback：在每個 epoch 計算 PSNR & IoU, 並在 epoch 5,10,20 儲存驗證影像 ──────────
class ValMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, val_files, epochs_to_save=[5,10,20]):
        super().__init__()
        self.val_ds = val_ds
        self.val_files = val_files
        self.epochs_to_save = epochs_to_save

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        # --- 定量指標：遍歷整個 validation set 計算 PSNR & IoU ---
        psnr_vals = []
        iou_vals  = []
        for imgs, masks in self.val_ds:
            preds = self.model.predict(imgs)
            pred_cls = tf.argmax(preds, axis=-1).numpy()
            gt_cls   = tf.argmax(masks, axis=-1).numpy()
            B, H, W = pred_cls.shape
            for b in range(B):
                pred_color = PALETTE[pred_cls[b]]
                gt_color   = PALETTE[gt_cls[b]]
                psnr_vals.append(compute_psnr(gt_color, pred_color, data_range=255))
                iou_vals.append(np.sum(pred_cls[b]==gt_cls[b]) / (H*W))
        mean_psnr = np.mean(psnr_vals)
        mean_iou  = np.mean(iou_vals)
        print(f"Epoch {ep}: Val Mean PSNR={mean_psnr:.2f}, Val Mean IoU={mean_iou:.3f}")

        # --- 定性結果：在特定 epoch 儲存三張驗證圖 ---
        if ep in self.epochs_to_save:
            imgs, masks = next(iter(self.val_ds))  # 取第一 batch
            preds = self.model.predict(imgs)
            for i in range(min(3, imgs.shape[0])):
                inp = (imgs[i].numpy() * 255).astype(np.uint8)
                pred_cls = tf.argmax(preds[i], axis=-1).numpy()
                gt_cls   = tf.argmax(masks[i], axis=-1).numpy()
                color_pred = PALETTE[pred_cls]
                color_gt   = PALETTE[gt_cls]
                fig, axs = plt.subplots(1,3, figsize=(9,3))
                axs[0].imshow(inp); axs[0].set_title('Input');  axs[0].axis('off')
                axs[1].imshow(color_pred); axs[1].set_title('Output'); axs[1].axis('off')
                axs[2].imshow(color_gt);   axs[2].set_title('GT');     axs[2].axis('off')
                fname = f"epoch{ep}_sample{i+1}.png"
                fig.savefig(os.path.join(VAL_DIR, fname), bbox_inches='tight', pad_inches=0)
                plt.close(fig)
            print(f"Saved Val Images for epoch {ep}")

# ─── 開始訓練 ─────────────────────────────────────────────────────
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb,
               ValMetricsCallback(val_ds, val_files)]
)


