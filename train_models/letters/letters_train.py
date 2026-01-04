

import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# -----------------------------
# 1) Config
# -----------------------------
imageSize = 64
train_dir = '/kaggle/input/synthetic-asl-alphabet/Train_Alphabet/'

classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', 'Blank'
]
label_map = {name: i for i, name in enumerate(classes)}
inv_label_map = {i: name for name, i in label_map.items()}

print("Num classes:", len(classes))


# -----------------------------
# 2) Load data (GRAYSCALE, 64x64, uint8)
# -----------------------------
def get_data_u8(folder):
    X = []
    y = []
    print("Starting full data import...")

    for folderName in classes:
        folder_path = os.path.join(folder, folderName)
        if not os.path.isdir(folder_path):
            print(f"Skipping {folderName} - folder not found.")
            continue

        label = label_map[folderName]
        print(f"Processing: {folderName} (Label {label})", end="\r")

        for image_filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (imageSize, imageSize), interpolation=cv2.INTER_AREA)
                X.append(img)       # (64,64) uint8
                y.append(label)
            except Exception:
                continue

    print(f"\nSuccessfully loaded {len(X)} images.")
    return np.array(X, dtype=np.uint8), np.array(y, dtype=np.int32)

X_u8, y = get_data_u8(train_dir)
print("X_u8:", X_u8.shape, X_u8.dtype)
print("y:", y.shape, y.dtype, "unique:", np.unique(y))



def show_examples_by_class_u8(X_u8, y, class_names=None, classes_idx=None, k=10, seed=42, title=""):
    rng = np.random.default_rng(seed)
    if classes_idx is None:
        classes_idx = np.unique(y)
    classes_idx = list(classes_idx)

    fig, axes = plt.subplots(len(classes_idx), k, figsize=(k*1.6, len(classes_idx)*1.6))
    if len(classes_idx) == 1:
        axes = np.array([axes])

    for r, cls in enumerate(classes_idx):
        idxs = np.where(y == cls)[0]
        pick = rng.choice(idxs, size=min(k, len(idxs)), replace=False)

        for j in range(k):
            ax = axes[r, j]
            if j < len(pick):
                gi = int(pick[j])
                ax.imshow(X_u8[gi], cmap="gray", vmin=0, vmax=255)
                name = class_names[int(cls)] if class_names is not None else str(int(cls))
                ax.set_title(f"{name}\nID={gi}", fontsize=8)
            ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

show_examples_by_class_u8(
    X_u8, y,
    class_names=[inv_label_map[i] for i in range(len(classes))],
    classes_idx=range(len(classes)),
    k=10,
    seed=123,
    title="10 exemple NEAUGMENTATE / clasă (GRAYSCALE)"
)


# -----------------------------
# Augumentation
# -----------------------------
def random_brightness_contrast(img, alpha_range=(0.6, 1.4), beta_range=(-50, 50)):
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def random_gamma(img, gamma_range=(0.7, 1.6)):
    gamma = random.uniform(*gamma_range)
    x = img.astype(np.float32) / 255.0
    x = np.power(x, gamma)
    return (x * 255.0).clip(0, 255).astype(np.uint8)

def random_affine(img, max_rotate=15, max_shift=0.08, scale_range=(0.9, 1.1)):
    h, w = img.shape
    angle = random.uniform(-max_rotate, max_rotate)
    scale = random.uniform(*scale_range)
    tx = random.uniform(-max_shift, max_shift) * w
    ty = random.uniform(-max_shift, max_shift) * h

    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def random_blur(img, p=0.25):
    if random.random() < p:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img

def random_noise(img, p=0.25, sigma_range=(3, 12)):
    if random.random() < p:
        sigma = random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return out.clip(0, 255).astype(np.uint8)
    return img

def augment_u8(img_u8):
    img = random_affine(img_u8)
    img = random_brightness_contrast(img)
    img = random_gamma(img)
    img = random_blur(img)
    img = random_noise(img)
    return img

def preprocess_u8_to_float(img_u8, use_clahe=True):
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_u8 = clahe.apply(img_u8)
    x = img_u8.astype(np.float32) / 255.0  # [0,1]
    return x[..., np.newaxis]              # (64,64,1)


def show_augmented_examples(X_u8, y, k=10, seed=777, title=""):
    rng = np.random.default_rng(seed)
    classes_idx = np.unique(y).tolist()

    fig, axes = plt.subplots(len(classes_idx), k, figsize=(k*1.6, len(classes_idx)*1.6))
    if len(classes_idx) == 1:
        axes = np.array([axes])

    for r, cls in enumerate(classes_idx):
        idxs = np.where(y == cls)[0]
        pick = rng.choice(idxs, size=min(k, len(idxs)), replace=False)

        for j in range(k):
            ax = axes[r, j]
            if j < len(pick):
                gi = int(pick[j])
                aug = augment_u8(X_u8[gi])
                ax.imshow(aug, cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{inv_label_map[int(cls)]}\naug", fontsize=8)
            ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

show_augmented_examples(
    X_u8, y,
    k=10,
    seed=999,
    title="10 exemple AUGMENTATE / clasă (OpenCV augment)"
)


# -----------------------------
# 6) Split train/val
# -----------------------------
from sklearn.model_selection import train_test_split
X_tr_u8, X_va_u8, y_tr, y_va = train_test_split(
    X_u8, y, test_size=0.30, random_state=42, stratify=y
)

print("Train:", X_tr_u8.shape, y_tr.shape)
print("Val:",   X_va_u8.shape, y_va.shape)


# -----------------------------
# 7) tf.data pipeline
#    - Train: augment -> preprocess (CLAHE + normalize) -> (64,64,1)
#    - Val:   preprocess only
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE
BATCH = 64

def tf_augment_and_preprocess(img_u8, label):
    # img_u8: tf.uint8 (64,64)
    # wrap OpenCV augment via tf.numpy_function
    def _aug_np(x):
        x = x.astype(np.uint8)
        x = augment_u8(x)
        x = preprocess_u8_to_float(x, use_clahe=True)  # float32 (64,64,1)
        return x

    x = tf.numpy_function(_aug_np, [img_u8], Tout=tf.float32)
    x.set_shape((imageSize, imageSize, 1))
    return x, label

def tf_preprocess_only(img_u8, label):
    def _prep_np(x):
        x = x.astype(np.uint8)
        x = preprocess_u8_to_float(x, use_clahe=True)
        return x

    x = tf.numpy_function(_prep_np, [img_u8], Tout=tf.float32)
    x.set_shape((imageSize, imageSize, 1))
    return x, label

train_ds = tf.data.Dataset.from_tensor_slices((X_tr_u8, y_tr))
train_ds = train_ds.shuffle(4096, seed=42, reshuffle_each_iteration=True)
train_ds = train_ds.map(tf_augment_and_preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_va_u8, y_va))
val_ds = val_ds.map(tf_preprocess_only, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH).prefetch(AUTOTUNE)


# -----------------------------
# Model (Keras CNN 64x64x1)
# -----------------------------
num_classes = len(classes)

model = keras.Sequential([
    layers.Input(shape=(imageSize, imageSize, 1)),

    layers.Conv2D(32, 3, padding="same"), layers.ReLU(),
    layers.MaxPool2D(),

    layers.Conv2D(64, 3, padding="same"), layers.ReLU(),
    layers.MaxPool2D(),

    layers.Conv2D(128, 3, padding="same"), layers.ReLU(),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Callbacks + Train + Save
# -----------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

ckpt_path = "/kaggle/working/asl_best.keras"
checkpoint = ModelCheckpoint(
    ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False, verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

# Save final model (.h5) + metadata
final_h5 = "/kaggle/working/ASL_gray_augmented.h5"
model.save(final_h5)
np.save("/kaggle/working/asl_label_map.npy", label_map, allow_pickle=True)

print("Saved best model to:", ckpt_path)
print("Saved final model to:", final_h5)
print("Saved label_map to: /kaggle/working/asl_label_map.npy")


# -----------------------------
# Plots (loss/acc)
# -----------------------------
import pandas as pd
metrics = pd.DataFrame(history.history)

metrics[["loss", "val_loss"]].plot()
plt.title("Loss")
plt.show()

metrics[["accuracy", "val_accuracy"]].plot()
plt.title("Accuracy")
plt.show()
