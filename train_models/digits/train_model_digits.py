import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt



def show_examples_by_class(X_u8, y_labels, classes=None, k=5, seed=42, title=""):
    rng = np.random.default_rng(seed)
    if classes is None:
        classes = np.unique(y_labels)
    classes = list(classes)

    fig, axes = plt.subplots(len(classes), k, figsize=(k * 2.2, len(classes) * 2.2))
    if len(classes) == 1:
        axes = np.array([axes])

    shown = []
    for r, cls in enumerate(classes):
        idxs = np.where(y_labels == cls)[0]
        if len(idxs) == 0:
            for j in range(k):
                axes[r, j].axis("off")
            continue

        pick = rng.choice(idxs, size=min(k, len(idxs)), replace=False)

        for j in range(k):
            ax = axes[r, j]
            if j < len(pick):
                gi = int(pick[j])
                img = X_u8[gi]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"class={int(cls)}\nID={gi}", fontsize=9)
                shown.append((gi, int(cls)))
            ax.axis("off")

    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    return shown


# -----------------------------
#  Load dataset
# -----------------------------
x_base = np.load('/kaggle/input/sign-language-digits-dataset/X.npy')
y_base = np.load('/kaggle/input/sign-language-digits-dataset/Y.npy')

print("X:", x_base.shape, x_base.dtype)
print("Y:", y_base.shape, y_base.dtype)

# Y can be one-hot (N,10) or labels (N,)
if y_base.ndim == 2 and y_base.shape[1] > 1:
    y_labels_full = np.argmax(y_base, axis=1).astype(np.int64)
else:
    y_labels_full = y_base.astype(np.int64)

# X can be (N,64,64) or (N,64,64,1) or (N,1,64,64)
X = x_base
if X.ndim == 4 and X.shape[-1] == 1:
    X = X[..., 0]
elif X.ndim == 4 and X.shape[1] == 1:
    X = X[:, 0, :, :]

# Ensure uint8 range
if X.max() <= 1.0:
    X_u8_full = (X * 255.0).clip(0, 255).astype(np.uint8)
else:
    X_u8_full = X.clip(0, 255).astype(np.uint8)

print("Full unique classes:", np.unique(y_labels_full))


# -----------------------------
# Remap
# -----------------------------
_ = show_examples_by_class(
    X_u8_full,
    y_labels_full,
    classes=sorted(np.unique(y_labels_full)),
    k=5,
    seed=123,
    title="PAS 1: 5 imagini NEAUGMENTATE din TOATE clasele originale"
)



class_remap = {0:3, 1:0, 3:3, 4:1, 6:4, 9:5, 7:3, 8:2, 2:3, 5:3}

orig_set = set(range(10))
assert set(class_remap.keys()) == orig_set, f"class_remap trebuie să aibă chei exact {sorted(orig_set)}"

y_new_raw = np.array([class_remap[int(y)] for y in y_labels_full], dtype=np.int64)

new_labels_sorted = sorted(np.unique(y_new_raw).tolist())
new_label_to_contig = {lab: i for i, lab in enumerate(new_labels_sorted)}
y_new = np.array([new_label_to_contig[int(v)] for v in y_new_raw], dtype=np.int64)

num_new_classes = len(new_labels_sorted)

print("Clase noi raw:", new_labels_sorted)
print("num_new_classes:", num_new_classes)
print("new_label_to_contig:", new_label_to_contig)
print("y_new unique:", np.unique(y_new))


_ = show_examples_by_class(
    X_u8_full,
    y_new,
    classes=range(num_new_classes),
    k=10,
    seed=999,
    title="PAS 2b: 10 imagini NEAUGMENTATE / clasă după UNIREA claselor"
)


# -----------------------------
# Augmentation + preprocess
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

def random_blur(img, p=0.2):
    if random.random() < p:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)
    return img

def random_noise(img, p=0.2, sigma_range=(3, 12)):
    if random.random() < p:
        sigma = random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return out.clip(0, 255).astype(np.uint8)
    return img

def augment(img_u8):
    img = random_affine(img_u8)
    img = random_brightness_contrast(img)
    img = random_gamma(img)
    img = random_blur(img)
    img = random_noise(img)
    return img

def preprocess(img_u8, use_clahe=True):
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_u8 = clahe.apply(img_u8)
    return (img_u8.astype(np.float32) / 255.0)


# -----------------------------
# Dataset
# -----------------------------
class DigitsDataset(Dataset):
    def __init__(self, X_u8, y, train=True):
        self.X = X_u8
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = int(self.y[idx])

        if self.train:
            img = augment(img)

        img = preprocess(img, use_clahe=True)
        img = torch.from_numpy(img).unsqueeze(0)
        return img, torch.tensor(label, dtype=torch.long)


# -----------------------------
#  Train/Val split
# -----------------------------
X_tr, X_va, y_tr, y_va = train_test_split(
    X_u8_full, y_new, test_size=0.15, random_state=42, stratify=y_new
)

train_ds = DigitsDataset(X_tr, y_tr, train=True)
val_ds   = DigitsDataset(X_va, y_va, train=False)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)


# -----------------------------
#  Model
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(num_classes=num_new_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


# -----------------------------
#  Train loop
# -----------------------------
@torch.no_grad()
def evaluate():
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    for x, y in val_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

best_acc = 0.0
EPOCHS = 25
save_path = "/kaggle/working/cnn_digits_merged_classes.pt"

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    seen = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * x.size(0)
        seen += x.size(0)

    train_loss = running / max(seen, 1)
    val_loss, val_acc = evaluate()
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(
            {
                "model_state": model.state_dict(),
                "num_new_classes": num_new_classes,
                "class_remap_original_to_new_raw": class_remap,
                "new_label_to_contig": new_label_to_contig,
            },
            save_path
        )

print("Best val acc:", best_acc)
print("Saved to:", save_path)
