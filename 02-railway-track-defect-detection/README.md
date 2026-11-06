# Railway Track Misalignment Detection with Fine-Tuned CNNs

Three ImageNet-pre-trained CNNs are fine-tuned to flag misaligned railway track from images in the TMD SIBGRAPI-2021 dataset (555 images), using one shared recipe so the architectures can be compared fairly.

## Approach
Partial fine-tuning (late blocks + new heads) of ResNet-50, EfficientNet-B3 and MobileNetV3-Large with AdamW, label smoothing, heavy augmentation and early stopping on a stratified 15 % validation split.

## Result
| Model | Test accuracy (111 images) | Missed defects |
|---|---|---|
| ResNet-50 | **96.40 %** | 2 / 55 |
| EfficientNet-B3 | 93.69 % | 5 / 55 |
| MobileNetV3-Large | 91.89 % | 9 / 55 |

## Files
| File | Contents |
|---|---|
| `finetune_track_misalignment.py` | data split, models, training, evaluation |
| `finetune_track_misalignment.ipynb` | notebook version |
| `report.pdf` | full report |
| `figures/` | curves, confusion matrices |

## Data
[TMD SIBGRAPI-2021](https://sites.google.com/site/koteswarraojerripothula/research-group/track-misalignment-detection-sibgrapi-2021) — place `TMD_SIBGRAP-2021.zip` in this folder.

## Run
```bash
pip install torch torchvision timm scikit-learn seaborn matplotlib
python finetune_track_misalignment.py
```

The full paper-style write-up — method, diagrams, results and discussion — is in [`report.pdf`](report.pdf).
