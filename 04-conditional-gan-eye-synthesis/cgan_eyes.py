# %% [markdown]
# # Conditional GAN for Synthetic Open / Closed Eye Images
# A label-conditioned DCGAN-style generator/discriminator pair is
# trained on the Blink Detection Dataset (50x50 grayscale; Closed = 0, Open = 1).
#
# Conditioning:
#   Generator     — learnable label embedding (2 -> 16) concatenated with noise z (100) -> 116-d input
#   Discriminator — learnable label embedding (2 -> 2500) reshaped to a 1x50x50 map and
#                   channel-concatenated with the image -> 2-channel input
#
# Expected data layout: DATA_ROOT/{Closed,Open}/*.png|jpg  (unzip "Blink Detection Dataset.zip")

# %%
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_ROOT = os.environ.get("BLINK_DATA", "./Blink Detection Dataset")
OUT_DIR = "./cgan_output"
IMG_SIZE = 50          # native resolution
NC = 1                 # grayscale
N_CLASSES = 2          # 0 = Closed, 1 = Open
LATENT_DIM = 100
EMB_DIM = 16           # generator class-embedding size
LR = 2e-4
BETA1 = 0.5
BATCH_SIZE = 128
N_EPOCHS = 100
SAVE_EVERY = 20        # save a progress grid every N epochs
N_GEN = 20             # images generated per class at the end
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for d in ["checkpoints", "generated/open", "generated/closed", "progress_grids"]:
    os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)
print(f"Device: {DEVICE}")


# %% Dataset
class BlinkDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.transform = [], transform
        for label, cls in enumerate(["Closed", "Open"]):
            folder = os.path.join(root, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, fname), label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return (self.transform(img) if self.transform else img), label


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),     # -> [-1, 1]
])
dataset = BlinkDataset(DATA_ROOT, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
print(f"Dataset size: {len(dataset):,} | Batches/epoch: {len(dataloader)}")


# %% Generator: [z ⊕ emb(y)] -> Linear -> 512x4x4 -> 4 ConvTranspose blocks -> 1x50x50
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, EMB_DIM)
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + EMB_DIM, 512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),   # 4 -> 8
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),   # 8 -> 16
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),    # 16 -> 32
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, NC, 4, 2, 1, bias=False),     # 32 -> 64
            nn.AdaptiveAvgPool2d(IMG_SIZE),                      # 64 -> 50
            nn.Tanh(),
        )

    def forward(self, z, labels):
        return self.net(torch.cat([z, self.label_emb(labels)], dim=1))


# %% Discriminator: [image ⊕ label map] (2x50x50) -> strided convs -> P(real | y)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, IMG_SIZE * IMG_SIZE)
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),                         # 25x25
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),   # 12x12
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),  # 6x6
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(-1, 1, IMG_SIZE, IMG_SIZE)
        return self.net(torch.cat([img, label_map], dim=1))


def weights_init(m):
    cls = m.__class__.__name__
    if "Conv" in cls:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in cls:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


G = Generator().to(DEVICE); G.apply(weights_init)
D = Discriminator().to(DEVICE); D.apply(weights_init)
print(f"G params: {sum(p.numel() for p in G.parameters()):,} | "
      f"D params: {sum(p.numel() for p in D.parameters()):,}")

opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
criterion = nn.BCELoss()

# fixed noise for the progress grid: 10 closed + 10 open
fixed_z = torch.randn(20, LATENT_DIM, device=DEVICE)
fixed_labels = torch.tensor([0] * 10 + [1] * 10, device=DEVICE)

# %% Training loop (one-sided label smoothing: real target = 0.9)
G_losses, D_losses = [], []
for epoch in range(1, N_EPOCHS + 1):
    epoch_g = epoch_d = 0.0
    t0 = time.time()
    for real_imgs, labels in dataloader:
        real_imgs, labels = real_imgs.to(DEVICE), labels.to(DEVICE)
        b = real_imgs.size(0)
        real_t = torch.full((b, 1), 0.9, device=DEVICE)
        fake_t = torch.zeros((b, 1), device=DEVICE)

        # --- Discriminator ---
        opt_D.zero_grad()
        loss_d_real = criterion(D(real_imgs, labels), real_t)
        z = torch.randn(b, LATENT_DIM, device=DEVICE)
        fake_imgs = G(z, labels)
        loss_d_fake = criterion(D(fake_imgs.detach(), labels), fake_t)
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        opt_D.step()

        # --- Generator (random target classes) ---
        opt_G.zero_grad()
        z = torch.randn(b, LATENT_DIM, device=DEVICE)
        rand_lbl = torch.randint(0, N_CLASSES, (b,), device=DEVICE)
        loss_g = criterion(D(G(z, rand_lbl), rand_lbl), torch.ones(b, 1, device=DEVICE))
        loss_g.backward()
        opt_G.step()

        epoch_g += loss_g.item()
        epoch_d += loss_d.item()

    G_losses.append(epoch_g / len(dataloader))
    D_losses.append(epoch_d / len(dataloader))
    print(f"Epoch [{epoch:3d}/{N_EPOCHS}] D_loss {D_losses[-1]:.4f} "
          f"G_loss {G_losses[-1]:.4f} ({time.time() - t0:.1f}s)")

    if epoch % SAVE_EVERY == 0 or epoch == 1:
        G.eval()
        with torch.no_grad():
            save_image(G(fixed_z, fixed_labels),
                       f"{OUT_DIR}/progress_grids/epoch_{epoch:03d}.png", nrow=10, normalize=True)
        G.train()

torch.save(G.state_dict(), f"{OUT_DIR}/checkpoints/G_final.pth")
torch.save(D.state_dict(), f"{OUT_DIR}/checkpoints/D_final.pth")

# %% Generate N_GEN images per class
G.eval()
with torch.no_grad():
    for cls_id, cls_name in [(0, "closed"), (1, "open")]:
        z = torch.randn(N_GEN, LATENT_DIM, device=DEVICE)
        lbl = torch.full((N_GEN,), cls_id, dtype=torch.long, device=DEVICE)
        imgs = G(z, lbl)                                     # (N, 1, 50, 50) in [-1, 1]
        save_image(imgs, f"{OUT_DIR}/generated/{cls_name}/grid_{cls_name}.png", nrow=5, normalize=True)
        imgs_np = ((imgs.cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        for i, arr in enumerate(imgs_np):
            Image.fromarray(arr[0], mode="L").save(f"{OUT_DIR}/generated/{cls_name}/{cls_name}_{i + 1:02d}.png")
        print(f"{N_GEN} {cls_name}-eye images saved.")

# %% Loss plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("cGAN Training Losses")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_losses.png", dpi=150)
print(f"All outputs in {OUT_DIR}")
