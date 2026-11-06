# %% [markdown]
# # Modified HoG (Roberts Cross) + LBP Global Feature + SVM — Cats vs Dogs
# Both descriptors are implemented from scratch (NumPy + cv2.filter2D);
# features are concatenated (1764 + 4096 = 5860-d) and classified with an RBF SVM.
#
# Data: loaded via `tensorflow_datasets` ("cats_vs_dogs"), as in the reported run.
# The Kaggle copy (https://www.kaggle.com/datasets/tongpython/cat-and-dog) also works:
# set DATA_SOURCE = "folder" and point DATA_DIR at a folder with `cats/` and `dogs/` sub-folders.
# A balanced 10,000-image subset is used (the full 25k set exhausted Colab free-tier RAM).

# %%
import gc
import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

DATA_SOURCE = "tfds"        # "tfds" or "folder"
DATA_DIR = "./cat-and-dog"  # used when DATA_SOURCE == "folder"
MAX_SAMPLES = 10000         # 5000 cats + 5000 dogs
TARGET_SIZE = (64, 64)
CELL_SIZE = 8
BLOCK_SIZE = 2
N_BINS = 9
GRID_Y, GRID_X = 4, 4       # LBP global grid
TEST_SIZE = 0.2
SVM_C = 10
RANDOM_STATE = 42

# %% Roberts Cross edge detector
ROBERTS_KX = np.array([[1, 0], [0, -1]], dtype=np.float32)
ROBERTS_KY = np.array([[0, 1], [-1, 0]], dtype=np.float32)


def roberts_cross(gray_img):
    """Return Gx, Gy, gradient magnitude and unsigned orientation (0–180°)."""
    gray = gray_img.astype(np.float32)
    gx = cv2.filter2D(gray, -1, ROBERTS_KX)
    gy = cv2.filter2D(gray, -1, ROBERTS_KY)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180
    return gx, gy, magnitude, angle


# %% Modified HoG (Roberts derivatives, 8x8 cells, 9 bins, 2x2 blocks, L2-Hys)
def hog_roberts(img):
    """64x64 image -> 1764-d descriptor (7x7 blocks x 36)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, _, magnitude, angle = roberts_cross(gray.astype(np.float32))
    H, W = gray.shape
    ncy, ncx = H // CELL_SIZE, W // CELL_SIZE
    bin_size = 180.0 / N_BINS

    cell_hists = np.zeros((ncy, ncx, N_BINS), dtype=np.float32)
    for cy in range(ncy):
        for cx in range(ncx):
            ys, xs = slice(cy * CELL_SIZE, (cy + 1) * CELL_SIZE), slice(cx * CELL_SIZE, (cx + 1) * CELL_SIZE)
            mag_cell, ang_cell = magnitude[ys, xs], angle[ys, xs]
            for b in range(N_BINS):
                mask = (ang_cell >= b * bin_size) & (ang_cell < (b + 1) * bin_size)
                cell_hists[cy, cx, b] = mag_cell[mask].sum()

    eps, descriptors = 1e-6, []
    for by in range(ncy - BLOCK_SIZE + 1):
        for bx in range(ncx - BLOCK_SIZE + 1):
            block = cell_hists[by:by + BLOCK_SIZE, bx:bx + BLOCK_SIZE, :].ravel()
            block = np.clip(block / (np.linalg.norm(block) + eps), 0, 0.2)   # L2-Hys
            descriptors.append(block / (np.linalg.norm(block) + eps))
    return np.concatenate(descriptors)


# %% LBP global feature (P=8, R=1; 4x4 grid of 256-bin histograms)
def compute_lbp_image(gray):
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    center = gray[1:-1, 1:-1].astype(np.int16)
    for p, (dy, dx) in enumerate(offsets):
        neighbour = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx].astype(np.int16)
        lbp[1:-1, 1:-1] += ((neighbour - center) >= 0).astype(np.uint8) * (2 ** p)
    return lbp


def lbp_global_feature(img):
    """64x64 image -> 4096-d descriptor (16 regions x 256 bins)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lbp_img = compute_lbp_image(gray)
    H, W = lbp_img.shape
    bh, bw = H // GRID_Y, W // GRID_X
    hists = []
    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            block = lbp_img[gy * bh:(gy + 1) * bh, gx * bw:(gx + 1) * bw]
            hist, _ = np.histogram(block.ravel(), bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hists.append(hist / (hist.sum() + 1e-6))
    return np.concatenate(hists)


def extract_features(img_bgr):
    """Resize -> [HoG(Roberts) | LBP] -> 5860-d vector."""
    img_r = cv2.resize(img_bgr, TARGET_SIZE)
    return np.concatenate([hog_roberts(img_r), lbp_global_feature(img_r)])


# %% Image iterator (balanced sampling, images discarded after feature extraction)
def iter_images():
    """Yield (BGR image, label) with 0 = cat, 1 = dog."""
    if DATA_SOURCE == "tfds":
        import tensorflow_datasets as tfds
        ds = tfds.load("cats_vs_dogs", split="train", as_supervised=True, shuffle_files=True)
        for img_t, lbl_t in ds.shuffle(buffer_size=10000, seed=RANDOM_STATE):
            yield cv2.cvtColor(img_t.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR), int(lbl_t.numpy())
    else:
        files = [(f, 0) for f in glob.glob(os.path.join(DATA_DIR, "**", "cats", "*.jpg"), recursive=True)]
        files += [(f, 1) for f in glob.glob(os.path.join(DATA_DIR, "**", "dogs", "*.jpg"), recursive=True)]
        np.random.default_rng(RANDOM_STATE).shuffle(files)
        for f, lbl in files:
            img = cv2.imread(f)
            if img is not None:
                yield img, lbl


# %% Feature extraction
per_class = MAX_SAMPLES // 2
counts, feats, labels = {0: 0, 1: 0}, [], []
with tqdm(total=MAX_SAMPLES, desc="Extracting features") as pbar:
    for img_bgr, lbl in iter_images():
        if counts[lbl] >= per_class:
            continue
        feats.append(extract_features(img_bgr))
        labels.append(lbl)
        counts[lbl] += 1
        pbar.update(1)
        if counts[0] >= per_class and counts[1] >= per_class:
            break

X = np.array(feats, dtype=np.float32)
y = np.array(labels)
del feats, labels; gc.collect()
print(f"Cats {counts[0]} | Dogs {counts[1]} | Feature matrix {X.shape} (~{X.nbytes / 1e6:.0f} MB)")
print("HoG (Roberts) 1764 + LBP global 4096 = 5860 dims")

# %% Split, scale, train SVM
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
del X, y; gc.collect()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training SVM (RBF, C=10, gamma='scale') ...")
svm = SVC(kernel="rbf", C=SVM_C, gamma="scale", random_state=RANDOM_STATE)
svm.fit(X_train, y_train)

# %% Evaluate
y_pred = svm.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["cat", "dog"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
plt.title("Confusion Matrix"); plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.show()

# %% Visualise Roberts gradients and LBP codes for one cat and one dog
samples = {}
for img_bgr, lbl in iter_images():
    samples.setdefault(lbl, img_bgr)
    if len(samples) == 2:
        break
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Roberts Cross Gradients & LBP | Row 0: Cat  Row 1: Dog", fontsize=13)
for row, (lbl, name) in enumerate([(0, "Cat"), (1, "Dog")]):
    img = cv2.resize(samples[lbl], TARGET_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, _, mag, ang = roberts_cross(gray)
    axes[row, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[row, 0].set_title(f"{name} – Original")
    axes[row, 1].imshow(mag, cmap="hot"); axes[row, 1].set_title("Roberts |G| Magnitude")
    axes[row, 2].imshow(ang, cmap="hsv"); axes[row, 2].set_title("Roberts Gradient Angle")
    axes[row, 3].imshow(compute_lbp_image(gray), cmap="gray"); axes[row, 3].set_title("LBP Code Image")
    for ax in axes[row]:
        ax.axis("off")
plt.tight_layout(); plt.savefig("hog_lbp_visualisation.png", dpi=120); plt.show()
