"""
Cervical-dystonia head-pose features from a pre-trained body-landmark model.

MediaPipe PoseLandmarker (Heavy) gives 33 body landmarks; five are used:
  0 nose, 7 left ear, 8 right ear, 11 left shoulder, 12 right shoulder.

Features (distances normalised by shoulder width for scale invariance)
  F1 head tilt angle  θ  : angle of (left ear − left shoulder) from vertical, degrees (supporting only)
  F2 nose midline offset Δx = (nose_x − shoulder_mid_x) / shoulder_width          -> rotational torticollis
  F3 ear height asymmetry Δy = (left_ear_y − right_ear_y) / shoulder_width          -> laterocollis
  F4 shoulder inclination φ  = atan2(−(LS_y − RS_y), LS_x − RS_x), degrees          -> laterocollis

Usage
  python dystonia_pose_features.py normal_1.jpg:Normal normal_2.jpg:Normal \
         rotational.jpg:"Rot. Torticollis" laterocollis.jpg:Laterocollis

Requirements: mediapipe==0.10.14 opencv-python numpy pandas matplotlib
"""
import os
import sys
import urllib.request

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
MODEL_PATH = "pose_landmarker_heavy.task"
OUT_DIR = "results"

# decision thresholds (from the rule table in the report)
T_F2, T_F3, T_F4 = 0.10, 0.08, 10.0


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose landmarker model (~30 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# --------------------------------------------------------------------------- features
def extract_features(landmarks, image_width, image_height):
    def xy(i):
        return np.array([landmarks[i].x * image_width, landmarks[i].y * image_height])

    nose, l_ear, r_ear, l_sh, r_sh = xy(0), xy(7), xy(8), xy(11), xy(12)

    # F1 — head tilt: ear-to-shoulder vector vs the upward vertical [0, -1]
    vec = l_ear - l_sh
    cos_t = np.dot(vec, [0, -1]) / (np.linalg.norm(vec) + 1e-9)
    theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
    if vec[0] > 0:
        theta = -theta

    # F2 — nose offset from the shoulder midline
    shoulder_mid = (l_sh + r_sh) / 2.0
    shoulder_w = np.linalg.norm(r_sh - l_sh) + 1e-9
    delta_x = (nose[0] - shoulder_mid[0]) / shoulder_w

    # F3 — ear height asymmetry (image y grows downward: +ve = left ear lower)
    delta_y = (l_ear[1] - r_ear[1]) / shoulder_w

    # F4 — shoulder line angle from horizontal
    sh_vec = l_sh - r_sh
    phi = np.degrees(np.arctan2(-sh_vec[1], sh_vec[0]))

    feats = {"F1_head_tilt_deg": round(float(theta), 2),
             "F2_nose_offset_norm": round(float(delta_x), 4),
             "F3_ear_asymmetry_norm": round(float(delta_y), 4),
             "F4_shoulder_incl_deg": round(float(phi), 2)}
    coords = dict(nose=nose, left_ear=l_ear, right_ear=r_ear, left_sh=l_sh, right_sh=r_sh,
                  shoulder_mid=shoulder_mid)
    return feats, coords


def classify_pose(f):
    """Rule table: F2 flags rotation, F3 + F4 flag lateral tilt; F1 is supporting only
    (it is inflated in frontal photographs by perspective)."""
    f2, f3, f4 = abs(f["F2_nose_offset_norm"]), abs(f["F3_ear_asymmetry_norm"]), abs(f["F4_shoulder_incl_deg"])
    if f2 < T_F2 and f3 < T_F3 and f4 < T_F4:
        return "Normal"
    if f3 > T_F3 and f4 > T_F4:
        return "Laterocollis"
    if f2 > T_F2 and f3 < T_F3:
        return "Rotational Torticollis"
    return "Abnormal (mixed/uncertain)"


def classify_pose_v0(f):
    """Initial rule (kept as a baseline for comparison). It required |F1| < 12°
    for Normal, which frontal photos never satisfy, so both normal images came out 'mixed'."""
    f1, f2, f3 = abs(f["F1_head_tilt_deg"]), abs(f["F2_nose_offset_norm"]), abs(f["F3_ear_asymmetry_norm"])
    if f1 < 12 and f2 < 0.10 and f3 < 0.08:
        return "Normal"
    if f3 > 0.08 and f1 > 12:
        return "Laterocollis"
    if f2 > 0.10:
        return "Rotational Torticollis"
    return "Abnormal (mixed/uncertain)"


# --------------------------------------------------------------------------- landmarking + drawing
def process_image(image_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"ERROR: could not read {image_path}")
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        output_segmentation_masks=False, num_poses=1,
        min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4, min_tracking_confidence=0.4)
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.pose_landmarks:
        print(f"No pose detected in {image_path}")
        return None, None

    lm = result.pose_landmarks[0]
    feats, c = extract_features(lm, w, h)

    ann = rgb.copy()
    for a, b in [(0, 7), (0, 8), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                 (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]:
        cv2.line(ann, (int(lm[a].x * w), int(lm[a].y * h)), (int(lm[b].x * w), int(lm[b].y * h)), (190, 190, 190), 1)
    pt = lambda p: (int(p[0]), int(p[1]))
    le, re, ls, rs, no, sm = (pt(c[k]) for k in ["left_ear", "right_ear", "left_sh", "right_sh", "nose", "shoulder_mid"])
    cv2.line(ann, le, ls, (255, 165, 0), 3)                 # F1
    cv2.line(ann, no, sm, (255, 60, 60), 2)                 # F2
    cv2.line(ann, re, (le[0], re[1]), (170, 100, 255), 1)   # F3 reference
    cv2.line(ann, (le[0], re[1]), le, (255, 60, 60), 2)     # F3 delta
    cv2.line(ann, ls, rs, (50, 210, 120), 3)                # F4
    for p, col in [(no, (255, 60, 60)), (le, (255, 165, 0)), (re, (130, 80, 255)),
                   (ls, (50, 210, 120)), (rs, (50, 210, 120)), (sm, (255, 255, 80))]:
        cv2.circle(ann, p, 8, col, -1); cv2.circle(ann, p, 8, (255, 255, 255), 1)
    return feats, ann


def show_result(img, feats, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img); axes[0].set_title(title, fontsize=13, fontweight="bold"); axes[0].axis("off")
    axes[0].legend(handles=[patches.Patch(color="orange", label="F1: ear–shoulder line"),
                            patches.Patch(color="#FF3C3C", label="F2/F3: nose offset / ear asymmetry"),
                            patches.Patch(color="#32D278", label="F4: shoulder line")],
                   loc="lower left", fontsize=8, framealpha=0.75, facecolor="#111", labelcolor="white")
    names = ["F1 Head Tilt\n(°)", "F2 Nose Offset\n(norm)", "F3 Ear Asym\n(norm)", "F4 Shoulder\nIncl (°)"]
    vals = list(feats.values())
    axes[1].bar(names, vals, color="#546E7A", edgecolor="white", width=0.5)
    axes[1].axhline(0, color="gray", lw=0.8)
    for i, v in enumerate(vals):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    axes[1].set_title("Feature values"); axes[1].set_ylabel("Value")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


def comparison_chart(df, save_path):
    cols = ["F1_head_tilt_deg", "F2_nose_offset_norm", "F3_ear_asymmetry_norm", "F4_shoulder_incl_deg"]
    titles = ["F1 Head Tilt (°)", "F2 Nose Offset", "F3 Ear Asymmetry", "F4 Shoulder Incl (°)"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, col, tit in zip(axes, cols, titles):
        ax.bar(df["Image"], df[col], color="#455A64")
        ax.axhline(0, color="gray", lw=0.8); ax.set_title(tit); ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Feature Comparison: Normal vs Cervical Dystonia", fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ensure_model()
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for arg in sys.argv[1:]:
        path, _, label = arg.partition(":")
        feats, ann = process_image(path)
        if feats is None:
            continue
        pred = classify_pose(feats)
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{name} [{label or '?'}] -> {pred}")
        for k, v in feats.items():
            print(f"  {k}: {v}")
        show_result(ann, feats, f"{label or name} | Predicted: {pred}", os.path.join(OUT_DIR, f"result_{name}.png"))
        rows.append({"Image": name, "True": label, "Predicted": pred, "Predicted (v0 rule)": classify_pose_v0(feats), **feats})
    if rows:
        df = pd.DataFrame(rows)
        print("\n" + df.to_string(index=False))
        df.to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
        comparison_chart(df, os.path.join(OUT_DIR, "comparison_chart.png"))
