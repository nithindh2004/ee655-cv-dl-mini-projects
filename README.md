# Computer Vision & Deep Learning Projects

Eight Mini projects in computer vision and deep learning: video analysis, transfer learning, CNN architecture design, generative models, pose-based feature engineering, hand-crafted descriptors, neural regression and classical image processing.

**Course:** EE655 Computer Vision & Deep Learning, IIT Kanpur · **Instructor:** Dr. Koteswar Rao Jerripothula

Each project folder contains runnable code (`.py` + `.ipynb`), a **paper-style PDF report** (abstract, method, experiments, discussion, references), its figures and a short README. Projects are ordered by scope, with the most complete data-driven systems first.


## Projects

| # | Project | Techniques | Headline result | Report |
|---|---|---|---|---|
| 01 | [Per-eye blink counter for patient video](01-blink-counter-video) | Haar / MediaPipe eye crops, CNN classifier, finite-state machine | Classifier ≈ 99.8 % val accuracy; 15 right-eye vs 0 left-eye blinks in 41 s | [PDF](01-blink-counter-video/report.pdf) |
| 02 | [Railway track misalignment detection](02-railway-track-defect-detection) | Transfer learning: ResNet-50, EfficientNet-B3, MobileNetV3 | ResNet-50 reaches 96.4 % test accuracy | [PDF](02-railway-track-defect-detection/report.pdf) |
| 03 | [AlexNet → fully convolutional AlexNet](03-alexnet-fully-convolutional) | FC → 1×1 conv, global average pooling, TinyImageNet | +2.3 pts top-1 with 62 % fewer parameters | [PDF](03-alexnet-fully-convolutional/report.pdf) |
| 04 | [Conditional GAN for eye synthesis](04-conditional-gan-eye-synthesis) | cGAN, label embeddings, spatial label maps | Open / closed eyes generated on demand | [PDF](04-conditional-gan-eye-synthesis/report.pdf) |
| 05 | [Cervical dystonia head-pose analysis](05-cervical-dystonia-pose-analysis) | MediaPipe Pose, geometric features, rule-based screening | Features separate rotated and tilted heads from normal ones by 7–28× | [PDF](05-cervical-dystonia-pose-analysis/report.pdf) |
| 06 | [HoG + LBP + SVM classifier](06-hog-lbp-svm-classifier) | From-scratch Roberts-cross HoG & regional LBP, RBF SVM | 78.05 % on Cats vs Dogs | [PDF](06-hog-lbp-svm-classifier/report.pdf) |
| 07 | [Neural complex-number calculator](07-neural-complex-calculator) | MLP regression, input/output standardisation, Huber loss | + / − within ~2.5 %, × 17.7 %, ÷ 32.2 % mean relative error | [PDF](07-neural-complex-calculator/report.pdf) |
| 08 | [Binary digit reader](08-binary-digit-reader) | Projection profiles, self-calibrating width rule | All 138 bits decoded correctly | [PDF](08-binary-digit-reader/report.pdf) |

## Repository layout

```
├── 01-blink-counter-video/               blink_counter.py · blink_counter.ipynb
├── 02-railway-track-defect-detection/    finetune_track_misalignment.py · .ipynb
├── 03-alexnet-fully-convolutional/       alexnet_cnn_vs_fcn.py · .ipynb
├── 04-conditional-gan-eye-synthesis/     cgan_eyes.py · .ipynb
├── 05-cervical-dystonia-pose-analysis/   dystonia_pose_features.py · .ipynb
├── 06-hog-lbp-svm-classifier/            hog_roberts_lbp_svm.py · .ipynb
├── 07-neural-complex-calculator/         calculator_v1.py · calculator_v2.py · complex_calculator.ipynb
├── 08-binary-digit-reader/               extract_binary.py · .ipynb · binary_image.png
├── requirements.txt
└── README.md
```

Every project folder also contains `report.pdf`, `figures/` and a `README.md`.

## Setup

```bash
git clone <this-repo-url>
cd EE655-Computer-Vision-Deep-Learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then follow the **Run** section of the project's README. Projects 01–04 were trained on a Google Colab T4 GPU. Projects 06–08 run on a CPU.

## Datasets

Datasets are not included; download them and place them as each project README describes.

| Project | Dataset |
|---|---|
| 01, 04 | Blink Detection Dataset: 41,649 open / closed eye images, 50 × 50 (plus a patient video for 01) |
| 02 | [Track Misalignment Detection, SIBGRAPI 2021](https://sites.google.com/site/koteswarraojerripothula/research-group/track-misalignment-detection-sibgrapi-2021) |
| 03 | [TinyImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) (200 classes, 64 × 64) |
| 05 | Any frontal upper-body photographs |
| 06 | Cats vs Dogs via `tensorflow_datasets`, or [Kaggle cat-and-dog](https://www.kaggle.com/datasets/tongpython/cat-and-dog) |
| 07, 08 | Synthetic data / included image |

## Reproducibility notes

- Reported numbers come from the original training runs. Where the released code differs from the code that produced them, the report says so: per-batch OneCycleLR stepping (03), the LBP histogram bin range (06), and the decision rule that now follows its own rule table (05).
- `calculator_v1.py` (07) was re-run on a CPU and reproduced the reported predictions to within about 1 %. `extract_binary.py` (08) reproduces its output exactly.

## Tech stack

Python · PyTorch · torchvision · timm · OpenCV · MediaPipe · scikit-learn · NumPy · Matplotlib · LaTeX

## Author

**Nithin D H** — B.Tech, IIT Kanpur
