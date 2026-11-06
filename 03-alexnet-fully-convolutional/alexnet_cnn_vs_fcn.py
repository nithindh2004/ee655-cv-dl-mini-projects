# %% [markdown]
# # AlexNet CNN vs AlexNet FCN on TinyImageNet
# The original AlexNet (Krizhevsky et al., 2012) is converted to a
# fully-convolutional network by replacing FC6/FC7/FC8 with convolutions and global average
# pooling. Both models are trained on TinyImageNet (200 classes) and compared.
#
# Dataset: https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet
# (the script can also download the official cs231n zip if `tiny-imagenet-200/` is absent).

# %%
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./tiny-imagenet-200"
NUM_CLASSES = 200
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3          # OneCycle peak LR
WD = 1e-4
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
print(f"Device: {DEVICE} | Classes: {NUM_CLASSES} | Epochs: {EPOCHS}")


# %% Dataset
def prepare_tiny(root=DATA_ROOT):
    """Download (if needed) and re-arrange the val split into ImageFolder layout."""
    root = Path(root)
    if not (root / "train").exists():
        import urllib.request, zipfile
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zp = root.parent / "tiny-imagenet-200.zip"
        urllib.request.urlretrieve(url, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(root.parent)
        zp.unlink()

    val_p = root / "val_proper"
    if not val_p.exists():
        val_p.mkdir()
        ann = {}
        with open(root / "val" / "val_annotations.txt") as f:
            for line in f:
                p = line.strip().split("\t")
                ann[p[0]] = p[1]
        for img, cls in ann.items():
            d = val_p / cls
            d.mkdir(exist_ok=True)
            s = root / "val" / "images" / img
            if s.exists():
                s.rename(d / img)
    return str(root / "train"), str(val_p)


def get_loaders():
    tr, vl = prepare_tiny()
    norm = T.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821])
    # Upsample 64x64 TinyImageNet to 227x227 so the original AlexNet needs no modification
    tr_tf = T.Compose([T.Resize(256), T.RandomCrop(227), T.RandomHorizontalFlip(),
                       T.ColorJitter(0.3, 0.3, 0.3, 0.1), T.ToTensor(), norm])
    vl_tf = T.Compose([T.Resize(256), T.CenterCrop(227), T.ToTensor(), norm])
    kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    return (DataLoader(datasets.ImageFolder(tr, tr_tf), shuffle=True, **kw),
            DataLoader(datasets.ImageFolder(vl, vl_tf), shuffle=False, **kw))


# %% Shared AlexNet backbone (Conv1-Conv5 + LRN + MaxPool), identical in both models
def alexnet_features():
    return nn.Sequential(
        nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),      # 55x55x96
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),                       # 27x27x96
        nn.Conv2d(96, 256, kernel_size=5, padding=2),                # 27x27x256
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),                       # 13x13x256
        nn.Conv2d(256, 384, kernel_size=3, padding=1),               # 13x13x384
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 384, kernel_size=3, padding=1),               # 13x13x384
        nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),               # 13x13x256
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),                       # 6x6x256
    )


# %% AlexNet CNN — original architecture with a 200-way FC8
class AlexNetCNN(nn.Module):
    """Input 227x227x3 -> features 6x6x256 -> flatten 9216 -> FC6 4096 -> FC7 4096 -> FC8 200."""

    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.features = alexnet_features()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),       # FC6
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),       # FC7
            nn.ReLU(inplace=True),
            nn.Linear(4096, nc),         # FC8
        )
        self._init_weights()

    def _init_weights(self):
        fc6, fc7 = self.classifier[1], self.classifier[4]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # original AlexNet: bias 1 for FC6/FC7, 0 elsewhere
                nn.init.constant_(m.bias, 1 if m is fc6 or m is fc7 else 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)          # 6x6x256 -> 9216
        return self.classifier(x)


# %% AlexNet FCN — same backbone, FC layers -> 1x1 convs + global average pooling
class AlexNetFCN(nn.Module):
    """
    Backbone identical to AlexNetCNN. No flatten.
      FC6 Linear(9216,4096) -> Conv2d(256, 4096, 1)
      FC7 Linear(4096,4096) -> Conv2d(4096, 4096, 1)
      FC8 Linear(4096, 200) -> Conv2d(4096, 200, 1)
    The 1x1 convs give a 6x6x200 class-score map; GlobalAvgPool -> 200 logits.
    Note: the *weight-equivalent* conversion of FC6 is a 6x6 conv (same 37.7M weights).
    Using a 1x1 conv for FC6 is a lighter design choice, not an exact equivalence.
    """

    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.features = alexnet_features()
        self.conv_fc = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 4096, kernel_size=1), nn.ReLU(inplace=True),    # Conv-FC6
            nn.Dropout2d(0.5),
            nn.Conv2d(4096, 4096, kernel_size=1), nn.ReLU(inplace=True),   # Conv-FC7
            nn.Conv2d(4096, nc, kernel_size=1),                            # Conv-score
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)             # (B, 256, 6, 6)
        x = self.conv_fc(x)              # (B, 200, 6, 6)
        x = self.global_pool(x)          # (B, 200, 1, 1)
        return x.flatten(1)              # (B, 200)


def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# %% Train / validation loops
def train_epoch(model, loader, crit, opt, sched):
    model.train()
    loss_sum = correct = total = 0
    for x, y in tqdm(loader, leave=False, desc=" train"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()                     # OneCycleLR is stepped once per batch
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def val_epoch(model, loader, crit):
    model.eval()
    loss_sum = c1 = c5 = total = 0
    for x, y in tqdm(loader, leave=False, desc=" val  "):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss_sum += crit(logits, y).item() * x.size(0)
        c1 += (logits.argmax(1) == y).sum().item()
        c5 += (logits.topk(5, 1).indices == y.unsqueeze(1)).any(1).sum().item()
        total += x.size(0)
    return loss_sum / total, 100.0 * c1 / total, 100.0 * c5 / total


def run(model, name, tr_loader, vl_loader):
    print(f"\n{'=' * 60}\nTraining {name} ({nparams(model) / 1e6:.2f}M params)\n{'=' * 60}")
    model = model.to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WD)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=EPOCHS,
                                          steps_per_epoch=len(tr_loader), pct_start=0.1)
    hist = {k: [] for k in ["tl", "vl", "ta", "va1", "va5", "sec"]}
    best = 0.0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, tr_loader, crit, opt, sched)
        vl, va1, va5 = val_epoch(model, vl_loader, crit)
        sec = time.time() - t0
        for k, v in zip(hist, [tl, vl, ta, va1, va5, sec]):
            hist[k].append(v)
        if va1 > best:
            best = va1
            torch.save(model.state_dict(), RESULTS / f"best_{name}.pth")
        print(f"Ep {ep:02d}/{EPOCHS} | TrLoss {tl:.3f} TrAcc {ta:.1f}% | "
              f"VlLoss {vl:.3f} Top-1 {va1:.1f}% Top-5 {va5:.1f}% | {sec:.1f}s")
    hist["best_top1"] = best
    json.dump(hist, open(RESULTS / f"hist_{name}.json", "w"), indent=2)
    print(f"\n{name} best Top-1: {best:.2f}%")
    return hist


# %% Plots
def plot(hc, hf):
    ep = range(1, EPOCHS + 1)
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
    ax = lambda pos: fig.add_subplot(gs[pos])

    def duo(a, key, yl, tit):
        a.plot(ep, hc[key], "o-", ms=3, color="#1a6eb5", label="CNN", lw=1.8)
        a.plot(ep, hf[key], "s-", ms=3, color="#e06c26", label="FCN", lw=1.8)
        a.set(xlabel="Epoch", ylabel=yl, title=tit); a.legend(fontsize=10)

    duo(ax((0, 0)), "tl", "Loss", "Training Loss")
    duo(ax((0, 1)), "vl", "Loss", "Validation Loss")
    a = ax((0, 2))
    a.plot(ep, hc["ta"], "o--", ms=3, color="#1a6eb5", alpha=0.5, lw=1.4, label="CNN train")
    a.plot(ep, hc["va1"], "o-", ms=3, color="#1a6eb5", lw=1.8, label="CNN val@1")
    a.plot(ep, hf["ta"], "s--", ms=3, color="#e06c26", alpha=0.5, lw=1.4, label="FCN train")
    a.plot(ep, hf["va1"], "s-", ms=3, color="#e06c26", lw=1.8, label="FCN val@1")
    a.set(xlabel="Epoch", ylabel="Acc (%)", title="Top-1 Accuracy"); a.legend(fontsize=9, ncol=2)
    duo(ax((1, 0)), "va5", "Top-5 Acc (%)", "Validation Top-5 Accuracy")

    a = ax((1, 1))
    mets = ["Top-1 (%)", "Top-5 (%)", "Params (M)"]
    cv = [hc["best_top1"], max(hc["va5"]), nparams(AlexNetCNN()) / 1e6]
    fv = [hf["best_top1"], max(hf["va5"]), nparams(AlexNetFCN()) / 1e6]
    x, w = np.arange(3), 0.35
    b1 = a.bar(x - w / 2, cv, w, label="CNN", color="#1a6eb5")
    b2 = a.bar(x + w / 2, fv, w, label="FCN", color="#e06c26")
    for b in [*b1, *b2]:
        a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{b.get_height():.1f}",
               ha="center", va="bottom", fontsize=8)
    a.set_xticks(x); a.set_xticklabels(mets, fontsize=9)
    a.set_title("Best Metrics Comparison"); a.legend(fontsize=10)
    duo(ax((1, 2)), "sec", "Seconds", "Epoch Training Time")

    fig.suptitle(f"AlexNet CNN vs AlexNet FCN — TinyImageNet ({NUM_CLASSES} classes, {EPOCHS} epochs)",
                 fontsize=14, y=1.01, fontweight="bold")
    fig.savefig(RESULTS / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def summary(hc, hf):
    cnn_p, fcn_p = nparams(AlexNetCNN()), nparams(AlexNetFCN())
    rows = [
        ("Best Val Top-1 Acc (%)", f"{hc['best_top1']:.2f}", f"{hf['best_top1']:.2f}"),
        ("Best Val Top-5 Acc (%)", f"{max(hc['va5']):.2f}", f"{max(hf['va5']):.2f}"),
        ("Final Train Acc (%)", f"{hc['ta'][-1]:.2f}", f"{hf['ta'][-1]:.2f}"),
        ("Trainable Params (M)", f"{cnn_p / 1e6:.2f}", f"{fcn_p / 1e6:.2f}"),
        ("Avg Epoch Time (s)", f"{np.mean(hc['sec']):.1f}", f"{np.mean(hf['sec']):.1f}"),
        ("FC layers", "3 (FC6/FC7/FC8)", "None (1x1 conv)"),
        ("Accepts variable input", "No", "Yes"),
    ]
    print("\n" + "=" * 65 + "\nQUANTITATIVE RESULTS SUMMARY\n" + "=" * 65)
    print(f"{'Metric':<32} {'AlexNet CNN':>14} {'AlexNet FCN':>14}\n" + "-" * 62)
    for r in rows:
        print(f"{r[0]:<32} {r[1]:>14} {r[2]:>14}")


# %% Main
if __name__ == "__main__":
    tr, vl = get_loaders()
    hc = run(AlexNetCNN(), "AlexNetCNN", tr, vl)
    hf = run(AlexNetFCN(), "AlexNetFCN", tr, vl)
    plot(hc, hf)
    summary(hc, hf)
