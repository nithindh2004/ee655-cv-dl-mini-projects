# Binary Digit Reader (Classical Image Processing)

A rule-based reader that decodes a 138-digit binary number from an image using projection profiles and a self-calibrating character-width threshold — no training data needed.

## Approach
Binarise → crop → horizontal projection (lines) → vertical projection (characters) → classify each character by width ('1' ≈ 11 px, '0' ≈ 22 px; threshold picked automatically at the largest width gap).

## Result
6 lines × 23 bits = 138 bits → decimal `232291987565042107861626583843617067324707` (verified by re-running on the image).

## Files
| File | Contents |
|---|---|
| `extract_binary.py` | the program |
| `extract_binary.ipynb` | notebook with outputs |
| `binary_image.png` | input image |
| `report.pdf` | full report |
| `figures/` | flowchart, projection profiles |

## Data
`binary_image.png` is included.

## Run
```bash
pip install numpy pillow
python extract_binary.py binary_image.png
```

The full paper-style write-up — method, diagrams, results and discussion — is in [`report.pdf`](report.pdf).
