# %% [markdown]
# # Binary Number Extraction from an Image
# Reads the binary digits printed in `binary_image.png`
# using projection profiles and a character-width rule ('1' is narrow, '0' is wide).
#
# Usage:  python extract_binary.py [path/to/image.png]

# %%
import sys

import numpy as np
from PIL import Image


def segment_runs(mask):
    """Return (start, end) index pairs of consecutive True runs in a 1-D boolean array."""
    runs, start = [], None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def extract_binary_from_image(image_path, verbose=True):
    img = np.array(Image.open(image_path).convert("L"))

    # 1) Binarise: dark background, light text
    binary = img > 128

    # 2) Crop to the bounding box of all text pixels
    rows = np.where(np.any(binary, axis=1))[0]
    cols = np.where(np.any(binary, axis=0))[0]
    cropped = binary[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # 3) Text lines from the horizontal projection (row sums)
    row_sums = cropped.sum(axis=1)
    line_segments = segment_runs(row_sums > cropped.shape[1] * 0.03)
    if verbose:
        print(f"Detected {len(line_segments)} text lines")

    # 4) Characters per line from the vertical projection (column sums)
    all_widths, per_line = [], []
    for s, e in line_segments:
        band = cropped[s:e, :]
        col_sums = band.sum(axis=0)
        chars = segment_runs(col_sums > band.shape[0] * 0.08)
        per_line.append(chars)
        all_widths += [c1 - c0 for c0, c1 in chars]

    # 5) Auto width threshold = midpoint of the largest gap between distinct widths
    widths = sorted(set(all_widths))
    gaps = [(widths[i + 1] - widths[i], i) for i in range(len(widths) - 1)]
    k = max(gaps)[1]
    width_threshold = (widths[k] + widths[k + 1]) / 2
    if verbose:
        print(f"Unique character widths: {widths}")
        print(f"Auto width threshold: {width_threshold:.1f}px  (narrower = '1', wider = '0')\n")

    # 6) Classify each character by width
    lines = ["".join("1" if (c1 - c0) < width_threshold else "0" for c0, c1 in chars)
             for chars in per_line]
    return lines, widths, width_threshold


# %%
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "binary_image.png"
    lines, _, _ = extract_binary_from_image(path)
    print("=== RESULT ===")
    for i, line in enumerate(lines, 1):
        print(f"Line {i}: {line}")
    full = "".join(lines)
    print(f"\nFull binary string ({len(full)} bits):\n{full}")
    print(f"\nDecimal value: {int(full, 2)}")
