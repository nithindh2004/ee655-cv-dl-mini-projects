# Roberts-Cross HoG + Global LBP + SVM (Cats vs Dogs)

A from-scratch hand-crafted pipeline for Cats vs Dogs: a HoG variant with Roberts-cross derivatives for shape, a regional LBP descriptor for texture, and an RBF SVM on the concatenated 5,860-d features.

## Approach
Roberts-cross gradients → 8×8 cells, 9 bins, 2×2 L2-Hys blocks (1,764-d) ⊕ LBP(8,1) codes in a 4×4 grid of 256-bin histograms (4,096-d) → StandardScaler → RBF SVM (C = 10).

## Result
**78.05 %** test accuracy on 2,000 held-out images (balanced 10k subset; precision = recall = 0.78 for both classes).

## Files
| File | Contents |
|---|---|
| `hog_roberts_lbp_svm.py` | features + SVM |
| `hog_roberts_lbp_svm.ipynb` | notebook version |
| `report.pdf` | full report |
| `figures/` | confusion matrix, gradient/LBP visualisation |

## Data
`tensorflow_datasets` *cats_vs_dogs* (downloaded automatically), or the [Kaggle copy](https://www.kaggle.com/datasets/tongpython/cat-and-dog) with `DATA_SOURCE = "folder"`.

## Run
```bash
pip install opencv-python scikit-learn seaborn tqdm tensorflow tensorflow-datasets
python hog_roberts_lbp_svm.py
```

The full paper-style write-up — method, diagrams, results and discussion — is in [`report.pdf`](report.pdf).
