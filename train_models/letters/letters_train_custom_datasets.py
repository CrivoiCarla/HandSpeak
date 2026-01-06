import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

# =============================
# 0) CONFIG
# =============================
DATA_DIR = r"./new_dataset"   # <-- pune aici folderul care are A,B,C,D,E,F
IMG_SIZE = 64
BATCH = 64
EPOCHS = 20
LR = 3e-4

CLASSES = ["A", "B", "C", "D", "E", "F"]
NUM_CLASSES = len(CLASSES)

# split ratios
TEST_SIZE = 0.15
VAL_SIZE_FROM_REMAINING = 0.15  # din ce rămâne după test (=> ~0.1275 din total)

# output files
OUT_DIR = DATA_DIR
BEST_PATH  = os.path.join(OUT_DIR, "asl_A_F_best.keras")
FINAL_PATH = os.path.join(OUT_DIR, "asl_A_F_final.h5")
MAP_PATH   = os.path.join(OUT_DIR, "asl_A_F_class_map.npy")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# =============================
# 1) GPU INFO
# =============================
print("TF version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU detected. Memory growth enabled.")
    except Exception as e:
        print("Could not set memory growth:", e)

# =============================
# 2) LOAD FILES FROM FOLDERS
# =============================
def load_from_single_folder(root_dir, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []

    for c in classes:
        class_dir = os.path.join(root_dir, c)
        if not os.path.isdir(class_dir):
            raise ValueError(f"Missing class folder: {class_dir}")

        for r, _, files in os.walk(class_dir):
            for fn in files:
                if fn.lower().endswith(IMG_EXTS):
                    paths.append(os.path.join(r, fn))
                    labels.append(class_to_idx[c])

    paths = np.array(paths, dtype=object)
    labels = np.array(labels, dtype=np.int32)

    if len(paths) == 0:
        raise ValueError(f"No images found under {root_dir} with extensions {IMG_EXTS}")

    # Shuffle once (deterministic)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(paths))
    return paths[perm], labels[perm], class_to_idx

all_paths, all_labels, class_to_idx = load_from_single_folder(DATA_DIR, CLASSES)
idx_to_class = {v: k for k, v in class_to_idx.items()}

print("Total images:", len(all_paths))
print("class_to_idx:", class_to_idx)
for c in range(NUM_CLASSES):
    print(f"  {idx_to_class[c]}: {(all_labels==c).sum()} images")

# =============================
# 3) SPLIT: TRAIN / VAL / TEST (stratified)
# =============================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    all_paths, all_labels,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=all_labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=VAL_SIZE_FROM_REMAINING,
    random_state=42,
    stratify=y_trainval
)

print("\nSplit sizes:")
print("Train:", len(X_train))
print("Val:  ", len(X_val))
print("Test: ", len(X_test))

# =============================
# 4) READ + PREPROCESS (OpenCV inside tf.data)
# =============================
def read_gray_resize(path, size=64):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # optional: CLAHE (comentat default; activează dacă vrei)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    img = img[..., np.newaxis]  # (64,64,1)
    return img

def tf_load_image(path, label):
    def _np_read(p):
        p = p.decode("utf-8")
        img = read_gray_resize(p, IMG_SIZE)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        return img

    x = tf.numpy_function(_np_read, [path], tf.float32)
    x.set_shape((IMG_SIZE, IMG_SIZE, 1))
    return x, label

AUTOTUNE = tf.data.AUTOTUNE

def make_ds(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(4096, seed=42, reshuffle_each_iteration=True)
    ds = ds.map(tf_load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(X_train, y_train, training=True)
val_ds   = make_ds(X_val,   y_val,   training=False)
test_ds  = make_ds(X_test,  y_test,  training=False)

# =============================
# 5) MODEL
# =============================
with tf.device("/GPU:0" if gpus else "/CPU:0"):
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

model.summary()

# =============================
# 6) TRAIN + SAVE
# =============================
early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint(BEST_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.4f}")

# Save final
model.save(FINAL_PATH)

# Save mapping
np.save(MAP_PATH, {i: idx_to_class[i] for i in range(NUM_CLASSES)}, allow_pickle=True)

print("\nSaved best model:", BEST_PATH)
print("Saved final model:", FINAL_PATH)
print("Saved class map:", MAP_PATH)
