# AlexNet → Fully Convolutional AlexNet on TinyImageNet

We convert AlexNet into a fully convolutional network (1×1 convolutions + global average pooling instead of FC layers) and train both from scratch on TinyImageNet (200 classes, 100k images) to compare accuracy, parameters and behaviour.

## Approach
Original AlexNet backbone on 227×227 upsampled inputs. FCN replaces FC6/FC7/FC8 with 1×1 convolutions and global average pooling (no flatten). Both trained 30 epochs with SGD + OneCycle.

## Result
| | AlexNet CNN | AlexNet FCN |
|---|---|---|
| Best val top-1 | 34.7 % | **37.0 %** |
| Best val top-5 | 60.2 % | **63.2 %** |
| Parameters | 59.10 M | **22.40 M** (−62 %) |

## Files
| File | Contents |
|---|---|
| `alexnet_cnn_vs_fcn.py` | models, training, plots |
| `alexnet_cnn_vs_fcn.ipynb` | notebook version |
| `report.pdf` | full report |
| `figures/` | architecture diagram, training curves |

## Data
[TinyImageNet (Kaggle)](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) — the script also downloads the official zip if missing.

## Run
```bash
pip install torch torchvision tqdm matplotlib
python alexnet_cnn_vs_fcn.py   # GPU recommended
```

The full paper-style write-up — method, diagrams, results and discussion — is in [`report.pdf`](report.pdf).
