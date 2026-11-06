# %% [markdown]
# # Track Misalignment Detection — Fine-tuning Three Pre-trained CNNs
# Binary classification (Defective vs Normal) on the
# TMD SIBGRAPI-2021 dataset with ResNet-50, EfficientNet-B3 and MobileNetV3-Large.
#
# Dataset: https://sites.google.com/site/koteswarraojerripothula/research-group/track-misalignment-detection-sibgrapi-2021
# Place `TMD_SIBGRAP-2021.zip` next to this script (expected layout inside: NewR22/{train,test}/{Defective,Normal}).
#
# pip install torch torchvision timm scikit-learn seaborn

# %%
import copy
import os
import random
import shutil
import time
import zipfile

import numpy as np
import matplotlib.pyplot as plt

ZIP_PATH = "TMD_SIBGRAP-2021.zip"
EXTRACT_ROOT = "./data_raw"
DATA_DIR = "./dataset"           # ImageFolder-compatible root built below
RANDOM_SEED = 42
VAL_FRACTION = 0.15              # fraction of each training class held out for validation
CLASSES = ["Defective", "Normal"]
SRC_TRAIN = os.path.join(EXTRACT_ROOT, "NewR22", "train")
SRC_TEST = os.path.join(EXTRACT_ROOT, "NewR22", "test")

if not os.path.exists(EXTRACT_ROOT):
    print("Extracting zip ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(EXTRACT_ROOT)

random.seed(RANDOM_SEED)


# %% Build train / val / test folders (stratified 15% val carved from train)
def build_split(src_dir, dst_root, split_name, val_frac=0.0):
    counts = {}
    for cls in CLASSES:
        src = os.path.join(src_dir, cls)
        dst = os.path.join(dst_root, split_name, cls)
        os.makedirs(dst, exist_ok=True)
        imgs = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(imgs)
        if val_frac > 0:
            n_val = max(1, int(len(imgs) * val_frac))
            val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]
            val_dst = os.path.join(dst_root, "val", cls)
            os.makedirs(val_dst, exist_ok=True)
            for f in val_imgs:
                shutil.copy2(os.path.join(src, f), os.path.join(val_dst, f))
            for f in train_imgs:
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            counts[cls] = {"train": len(train_imgs), "val": len(val_imgs)}
        else:
            for f in imgs:
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            counts[cls] = len(imgs)
    return counts


if not os.path.exists(DATA_DIR):
    train_counts = build_split(SRC_TRAIN, DATA_DIR, "train", val_frac=VAL_FRACTION)
    test_counts = build_split(SRC_TEST, DATA_DIR, "test")
    for cls in CLASSES:
        t = train_counts[cls]
        print(f" {cls}: train={t['train']} val={t['val']} test={test_counts[cls]}")
else:
    print("Dataset folder already exists — skipping rebuild.")

# %% Configuration
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm


class Config:
    DATA_DIR = DATA_DIR
    OUTPUT_DIR = "./outputs"
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LR = 3e-4
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    WEIGHT_DECAY = 1e-4
    EARLY_STOP = 10
    IMG_SIZE = 224
    NUM_CLASSES = 2
    DROPOUT = 0.4
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    WORKERS = 2


cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
random.seed(cfg.SEED); np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)
print(f"Device: {cfg.DEVICE} | PyTorch {torch.__version__}")

# %% Transforms & data loaders (augmentation on train only)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE + 32, cfg.IMG_SIZE + 32)),
    transforms.RandomCrop(cfg.IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def get_loaders():
    train_ds = datasets.ImageFolder(os.path.join(cfg.DATA_DIR, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(cfg.DATA_DIR, "val"), val_tf)
    test_ds = datasets.ImageFolder(os.path.join(cfg.DATA_DIR, "test"), val_tf)
    kw = dict(batch_size=cfg.BATCH_SIZE, num_workers=cfg.WORKERS, pin_memory=True)
    print(f"Classes: {train_ds.classes} | Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")
    return (DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw), train_ds.classes)


# %% Models (partial fine-tuning: early layers frozen, late layers + new head trained)
def build_resnet50(num_classes, dropout):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for name, p in model.named_parameters():
        p.requires_grad = any(s in name for s in ["layer3", "layer4", "fc"])
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


class EfficientNetWrapper(nn.Module):
    def __init__(self, backbone, num_classes, dropout):
        super().__init__()
        self.bb = backbone
        in_feat = backbone.num_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.BatchNorm1d(in_feat), nn.Dropout(dropout),
            nn.Linear(in_feat, 256), nn.ReLU(),
            nn.Dropout(dropout / 2), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.bb.forward_features(x))


def build_efficientnet_b3(num_classes, dropout):
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0)
    blocks = list(model.blocks.children())
    for blk in blocks[:-2]:
        for p in blk.parameters():
            p.requires_grad = False
    for blk in blocks[-2:]:
        for p in blk.parameters():
            p.requires_grad = True
    return EfficientNetWrapper(model, num_classes, dropout)


def build_mobilenet_v3(num_classes, dropout):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    for layer in list(model.features.children())[:-3]:
        for p in layer.parameters():
            p.requires_grad = False
    in_feat = model.classifier[0].in_features
    model.classifier = nn.Sequential(nn.Linear(in_feat, 256), nn.Hardswish(),
                                     nn.Dropout(dropout), nn.Linear(256, num_classes))
    return model


MODEL_REGISTRY = {
    "ResNet50": build_resnet50,
    "EfficientNet-B3": build_efficientnet_b3,
    "MobileNetV3": build_mobilenet_v3,
}


# %% Training & evaluation helpers
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        # NOTE: train accuracy uses a second forward pass after the update (as in the run
        # that produced the reported results). It slightly inflates train accuracy and also
        # updates BatchNorm running stats a second time; reuse the first logits if you
        # want a strictly standard loop.
        correct += (model(imgs).argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    preds_all, labels_all = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        total_loss += criterion(out, labels).item() * imgs.size(0)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, preds_all, labels_all


def train_model(name, model, train_loader, val_loader):
    model = model.to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                     patience=cfg.LR_PATIENCE, factor=cfg.LR_FACTOR)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, no_improve, t0 = 0.0, None, 0, time.time()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 55}\n {name} ({trainable:,} / {total_p:,} trainable params)\n{'=' * 55}")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE)
        vl, va, _, _ = evaluate(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step(va)
        for k, v in zip(history, [tl, vl, ta, va]):
            history[k].append(v)
        flag = ""
        if va > best_acc:
            best_acc, best_state, no_improve, flag = va, copy.deepcopy(model.state_dict()), 0, " *"
        else:
            no_improve += 1
        print(f" Ep {epoch:3d}/{cfg.NUM_EPOCHS} | Train {ta:.4f} | Val {va:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.1e}{flag}")
        if no_improve >= cfg.EARLY_STOP:
            print(f" Early stop at epoch {epoch}.")
            break

    print(f" Best Val Acc: {best_acc:.4f} | Time: {(time.time() - t0) / 60:.1f} min")
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(cfg.OUTPUT_DIR, f"{name.replace('-', '_')}_best.pth"))
    return model, history, best_acc


# %% Plotting & test evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def plot_history(name, history):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{name} — Training Curves", fontsize=13)
    axes[0].plot(epochs, history["train_acc"], label="Train")
    axes[0].plot(epochs, history["val_acc"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, history["train_loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"{name.replace('-', '_')}_curves.png"), dpi=120)
    plt.show()


def full_evaluation(name, model, test_loader, class_names):
    _, acc, preds, labels = evaluate(model, test_loader, nn.CrossEntropyLoss(), cfg.DEVICE)
    print(f"\n[{name}] Test Acc: {acc:.4f}")
    print(classification_report(labels, preds, target_names=class_names))
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} — Confusion Matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"{name.replace('-', '_')}_cm.png"), dpi=120)
    plt.show()
    return acc


def plot_comparison(results):
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black")
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{acc:.2f}%",
                 ha="center", va="bottom", fontsize=11)
    plt.ylim(0, 108); plt.ylabel("Test Accuracy (%)")
    plt.title("Model Comparison — Track Misalignment Detection")
    plt.tight_layout(); plt.savefig(os.path.join(cfg.OUTPUT_DIR, "comparison.png"), dpi=120)
    plt.show()


# %% Run all three models
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_loaders()
    results, all_history = {}, {}
    for model_name, builder in MODEL_REGISTRY.items():
        model = builder(cfg.NUM_CLASSES, cfg.DROPOUT)
        trained, history, _ = train_model(model_name, model, train_loader, val_loader)
        plot_history(model_name, history)
        results[model_name] = full_evaluation(model_name, trained, test_loader, class_names)
        all_history[model_name] = history

    print("\n" + "=" * 45 + "\n FINAL TEST RESULTS\n" + "=" * 45)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f" {name:<20} {acc * 100:.2f}%")
    plot_comparison(results)
