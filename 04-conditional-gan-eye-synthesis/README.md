# Conditional GAN for Open / Closed Eye Synthesis

A conditional GAN trained on 41,649 eye images that generates an open or a closed eye on request, with the class label injected into both the generator and the discriminator.

## Approach
DCGAN-style cGAN. **G:** label embedding (2→16) concatenated with 100-d noise. **D:** label embedding (2→2500) reshaped to a 50×50 map and stacked with the image as a second channel. BCE, one-sided label smoothing, 100 epochs.

## Result
20 generated images per class; flipping only the label on fixed noise switches closed lids ↔ open eyes with irises.

## Files
| File | Contents |
|---|---|
| `cgan_eyes.py` | training + generation |
| `cgan_eyes.ipynb` | notebook version |
| `report.pdf` | full report |
| `figures/` | architecture, loss curve, progress grids, generated samples |

## Data
Blink Detection Dataset (open/closed eye images) — unzip so `Blink Detection Dataset/{Closed,Open}/` exists.

## Run
```bash
pip install torch torchvision pillow matplotlib
python cgan_eyes.py
```

The full paper-style write-up — method, diagrams, results and discussion — is in [`report.pdf`](report.pdf).
