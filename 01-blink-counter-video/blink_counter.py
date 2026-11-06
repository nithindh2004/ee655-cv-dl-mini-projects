"""
Per-eye blink counter for a patient video.

Pipeline
  1. Train a small CNN (open vs closed) on the Blink Detection Dataset.
  2. For every video frame, crop each eye with a pre-trained detector:
       --detector mediapipe : MediaPipe FaceMesh landmarks (468-point facial landmark model)
       --detector haar      : OpenCV Haar cascades (frontal face, then eye within each face half)
  3. Classify each crop (P(closed) > 0.5 -> closed).
  4. A per-eye finite-state machine turns the closed/open sequence into blink events
     (a blink needs >= MIN_CLOSED_FRAMES consecutive closed frames).

"Left"/"Right" refer to the image halves as seen by the viewer, so "Left Eye" is the
patient's anatomical right eye. Check the annotated output video before quoting which
anatomical eye an asymmetric result belongs to.

Usage
  python blink_counter.py --data "Blink Detection Dataset" --video patient.mp4 --detector haar
  python blink_counter.py --skip-train --weights blink_output/best_eye_classifier.pth ...

Requirements: torch torchvision opencv-python mediapipe pillow matplotlib
"""
import argparse
import os
import random
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

IMG_SIZE = 50
BATCH_SIZE = 64
NUM_EPOCHS = 15
LR = 1e-3
MIN_CLOSED_FRAMES = 2      # eye must be closed for >= 2 consecutive frames to count as a blink
CLOSED_THRESHOLD = 0.5     # P(closed) above this -> closed
EYE_PADDING = 0.35
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# --------------------------------------------------------------------------- transforms
_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(), _norm,
])
eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), _norm])


# --------------------------------------------------------------------------- data
class BlinkDataset(Dataset):
    """DATA_DIR/{Closed,Open}/* ; label 0 = Closed, 1 = Open. Grayscale images loaded as RGB."""

    def __init__(self, root):
        self.samples = []
        for label, cls in enumerate(["Closed", "Open"]):
            folder = os.path.join(root, cls)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Expected folder '{folder}'")
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, fname), label))
        random.shuffle(self.samples)
        n_closed = sum(1 for _, l in self.samples if l == 0)
        print(f"[DATA] {len(self.samples):,} images (Closed={n_closed:,}, Open={len(self.samples) - n_closed:,})")

    def __len__(self):
        return len(self.samples)


class TransformSubset(Dataset):
    def __init__(self, subset, tf):
        self.subset, self.tf = subset, tf

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        path, label = self.subset.dataset.samples[self.subset.indices[i]]
        return self.tf(Image.open(path).convert("RGB")), label


# --------------------------------------------------------------------------- classifier
class EyeClassifier(nn.Module):
    """3 conv blocks (32/64/128, BN, ReLU) + FC 2048->256->2. 618,754 parameters."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),    # 25x25
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),   # 12x12
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(4),                                                             # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(True), nn.Dropout(0.4), nn.Linear(256, 2))

    def forward(self, x):
        return self.classifier(self.features(x))


def train_classifier(data_dir, out_dir):
    print("\n" + "=" * 60 + "\n PHASE 1: TRAINING EYE CLASSIFIER\n" + "=" * 60)
    full = BlinkDataset(data_dir)
    n_val = int(0.15 * len(full))
    tr, va = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(TransformSubset(tr, train_tf), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(TransformSubset(va, eval_tf), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"[DATA] Train {len(tr):,} | Val {n_val:,}")

    model = EyeClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_acc, best_path = 0.0, os.path.join(out_dir, "best_eye_classifier.pth")
    train_losses, val_accs = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_acc = 100.0 * correct / total
        train_losses.append(running / len(train_loader))
        val_accs.append(val_acc)
        print(f" Epoch [{epoch:2d}/{NUM_EPOCHS}] loss {train_losses[-1]:.4f} | val acc {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(train_losses, "b-o", ms=4, label="Train Loss"); a1.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
    a2.plot(val_accs, "g-o", ms=4, label="Val Accuracy"); a2.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Validation Accuracy")
    for a in (a1, a2):
        a.grid(True); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150); plt.close()

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print(f"[INFO] Best val accuracy {best_acc:.2f}% -> {best_path}")
    return model, best_acc


# --------------------------------------------------------------------------- eye croppers
def _pad_box(x1, y1, x2, y2, w, h, pad=EYE_PADDING, vpad_scale=1.0):
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - bw * pad)), min(w, int(x2 + bw * pad))
    y1, y2 = max(0, int(y1 - bh * pad * vpad_scale)), min(h, int(y2 + bh * pad * vpad_scale))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class MediaPipeEyeCropper:
    """Eye crops from MediaPipe FaceMesh eye-contour landmarks."""
    # contour of the eye on the image-left side (patient's right eye) and image-right side
    LEFT_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def __init__(self):
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __call__(self, frame):
        h, w = frame.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return False, None, None
        lm = res.multi_face_landmarks[0].landmark
        boxes = []
        for idx in (self.LEFT_IDX, self.RIGHT_IDX):
            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idx])
            boxes.append(_pad_box(*pts.min(0), *pts.max(0), w, h, vpad_scale=1.5))
        return True, boxes[0], boxes[1]

    def close(self):
        self.mesh.close()


class HaarEyeCropper:
    """Face via haarcascade_frontalface_default; eyes via haarcascade_eye inside each half of
    the upper 55% of the face box; 35% padding around the detected eye box."""

    def __init__(self):
        root = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(os.path.join(root, "haarcascade_frontalface_default.xml"))
        self.eye = cv2.CascadeClassifier(os.path.join(root, "haarcascade_eye.xml"))
        if self.face.empty() or self.eye.empty():
            raise RuntimeError(f"Haar cascade XMLs not found in {root}; install opencv-python 4.x")

    def __call__(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return False, None, None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        top_h = int(fh * 0.55)
        boxes = []
        for x0 in (fx, fx + fw // 2):                      # image-left half, image-right half
            roi = gray[fy:fy + top_h, x0:x0 + fw // 2]
            eyes = self.eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
            if len(eyes) == 0:
                boxes.append(None)
                continue
            ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])
            boxes.append(_pad_box(x0 + ex, fy + ey, x0 + ex + ew, fy + ey + eh, w, h))
        return True, boxes[0], boxes[1]

    def close(self):
        pass


# --------------------------------------------------------------------------- blink FSM
class BlinkCounter:
    def __init__(self, min_closed=MIN_CLOSED_FRAMES):
        self.state, self.blink_count, self.closed_frames, self.min_closed = "OPEN", 0, 0, min_closed
        self.p_closed, self.history_blink = [], []     # per frame (NaN when the eye was not found)
        self.detected = self.n_closed = 0

    def update(self, p_closed):
        if p_closed is None:                            # eye not detected: no state change
            self.p_closed.append(np.nan)
            self.history_blink.append(self.blink_count)
            return
        self.detected += 1
        self.p_closed.append(p_closed)
        closed = p_closed > CLOSED_THRESHOLD
        self.n_closed += int(closed)
        if self.state == "OPEN" and closed:
            self.state, self.closed_frames = "CLOSED", 1
        elif self.state == "CLOSED":
            if closed:
                self.closed_frames += 1
            else:
                if self.closed_frames >= self.min_closed:
                    self.blink_count += 1
                self.state, self.closed_frames = "OPEN", 0
        self.history_blink.append(self.blink_count)


@torch.no_grad()
def p_closed_of(model, crop_bgr):
    x = eval_tf(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    return torch.softmax(model(x), dim=1)[0, 0].item()


# --------------------------------------------------------------------------- video
def process_video(model, video_path, out_dir, detector):
    print("\n" + "=" * 60 + f"\n PHASE 2: PROCESSING VIDEO ({detector})\n" + "=" * 60)
    cropper = MediaPipeEyeCropper() if detector == "mediapipe" else HaarEyeCropper()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[VIDEO] {vw}x{vh} | {fps:.1f} fps | {total} frames | {total / fps:.1f}s")

    writer = cv2.VideoWriter(os.path.join(out_dir, "patient_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    left, right = BlinkCounter(), BlinkCounter()
    samples = {"open": [], "closed": []}
    model.eval()
    frame_idx = face_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        has_face, lbox, rbox = cropper(frame)
        face_frames += int(has_face)
        annotated = frame.copy()
        for box, ctr, side in [(lbox, left, "L"), (rbox, right, "R")]:
            p = None
            if box is not None:
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]
                p = p_closed_of(model, crop)
                key = "closed" if p > CLOSED_THRESHOLD else "open"
                if side == "R" and len(samples[key]) < 5:
                    samples[key].append(crop.copy())
                col = (0, 0, 220) if key == "closed" else (0, 200, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
                cv2.putText(annotated, f"{side}:{key.upper()} {p:.2f}", (x1, max(10, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            ctr.update(p)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (260, 110), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        for txt, y, col, sc in [("BLINK COUNTER", 20, (255, 255, 100), 0.55),
                                (f"Left Eye Blinks: {left.blink_count}", 45, (100, 230, 100), 0.5),
                                (f"Right Eye Blinks: {right.blink_count}", 70, (100, 180, 255), 0.5),
                                (f"Frame: {frame_idx}/{total}", 95, (180, 180, 180), 0.4)]:
            cv2.putText(annotated, txt, (8, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        writer.write(annotated)
        if frame_idx % 100 == 0:
            print(f" frame {frame_idx}/{total} | L {left.blink_count} R {right.blink_count}")

    cap.release(); writer.release(); cropper.close()
    return left, right, fps, frame_idx, face_frames, samples


# --------------------------------------------------------------------------- plots & report
def make_plots(left, right, fps, out_dir, samples):
    t = np.arange(len(left.p_closed)) / fps
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle("Blink Detection Results — Patient Video", fontsize=14, fontweight="bold")
    axes[0].plot(t, left.history_blink, color="#2ecc71", lw=2, label=f"Left Eye ({left.blink_count} blinks)")
    axes[0].plot(t, right.history_blink, color="#3498db", lw=2, label=f"Right Eye ({right.blink_count} blinks)")
    axes[0].set(ylabel="Cumulative Blinks", title="Blink Count Over Time"); axes[0].legend(); axes[0].grid(alpha=0.4)
    for ax, ctr, col, name in [(axes[1], left, "#e74c3c", "Left"), (axes[2], right, "#3498db", "Right")]:
        p = np.array(ctr.p_closed, dtype=float)
        ax.plot(t, p, color=col, lw=0.8, label="P(Closed)")
        ax.axhline(CLOSED_THRESHOLD, color="k", ls="--", lw=1, label=f"Threshold={CLOSED_THRESHOLD}")
        ax.fill_between(t, CLOSED_THRESHOLD, p, where=p > CLOSED_THRESHOLD, color=col, alpha=0.4, label="Closed region")
        ax.set(ylim=(0, 1.05), ylabel="P(Closed)", title=f"{name} Eye — Classifier Confidence (Closed)")
        ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    axes[2].set_xlabel("Time (seconds)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "blink_timeline.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = [left.blink_count, right.blink_count]
    bars = ax.bar(["Left Eye", "Right Eye"], vals, color=["#2ecc71", "#3498db"], edgecolor="black", width=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=14, fontweight="bold")
    ax.set(ylabel="Blink Count", title="Total Blinks — Left vs Right Eye", ylim=(0, max(vals) + 5))
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "blink_summary.png"), dpi=150); plt.close()

    if samples["open"] or samples["closed"]:
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle("Sample Eye Crops (Top: Open, Bottom: Closed)", fontsize=12, fontweight="bold")
        for row, key in enumerate(["open", "closed"]):
            for i in range(5):
                axes[row][i].axis("off")
                if i < len(samples[key]):
                    axes[row][i].imshow(cv2.cvtColor(samples[key][i], cv2.COLOR_BGR2GRAY), cmap="gray")
                    axes[row][i].set_title(key.capitalize(), fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sample_eye_crops.png"), dpi=150); plt.close()


def print_report(left, right, fps, n_frames, face_frames, val_acc):
    dur = n_frames / fps
    print("\n" + "=" * 60 + "\n BLINK DETECTION — FINAL REPORT\n" + "=" * 60)
    print(f" Video: {dur:.1f}s, {n_frames} frames @ {fps:.1f} fps | face found in {face_frames} frames")
    if val_acc is not None:
        print(f" Classifier best val accuracy: {val_acc:.2f}%")
    print(f" {'Metric':<26}{'Left Eye':>12}{'Right Eye':>12}")
    for name, a, b in [("Blink count", left.blink_count, right.blink_count),
                       ("Blink rate (/min)", f"{left.blink_count / dur * 60:.1f}", f"{right.blink_count / dur * 60:.1f}"),
                       ("Frames eye detected", left.detected, right.detected),
                       ("Frames classified closed", left.n_closed, right.n_closed),
                       ("Frames eye undetected", n_frames - left.detected, n_frames - right.detected)]:
        print(f" {name:<26}{a:>12}{b:>12}")
    print(" Typical resting blink rate: ~12–20 blinks/min (reference)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Blink Detection Dataset")
    ap.add_argument("--video", default="patient.mp4")
    ap.add_argument("--out", default="blink_output")
    ap.add_argument("--detector", choices=["haar", "mediapipe"], default="haar")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    if args.skip_train:
        model = EyeClassifier().to(DEVICE)
        model.load_state_dict(torch.load(args.weights or os.path.join(args.out, "best_eye_classifier.pth"),
                                         map_location=DEVICE))
        val_acc = None
    else:
        model, val_acc = train_classifier(args.data, args.out)
    left, right, fps, n, faces, samples = process_video(model, args.video, args.out, args.detector)
    make_plots(left, right, fps, args.out, samples)
    print_report(left, right, fps, n, faces, val_acc)
    print(f"\n[DONE] {(time.time() - t0) / 60:.1f} min — outputs in {args.out}/")
