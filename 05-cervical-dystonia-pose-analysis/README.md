# Cervical Dystonia Head-Pose Analysis from Body Landmarks

Interpretable geometric features from a pre-trained body-landmark model (MediaPipe Pose) that flag the abnormal head postures of cervical dystonia — rotation and lateral tilt — from a single frontal photo.

## Approach
MediaPipe PoseLandmarker (Heavy) → four shoulder-width-normalised features: F1 head-tilt angle (supporting), F2 nose midline offset (rotation), F3 ear height asymmetry and F4 shoulder inclination (lateral tilt) → rule table.

## Result
| Image | F2 | F3 | F4 (°) | Predicted |
|---|---|---|---|---|
| Normal 1 | −0.008 | 0.010 | 1.35 | Normal |
| Normal 2 | −0.051 | 0.007 | 1.92 | Normal |
| Rotational torticollis | **−0.359** | 0.020 | −0.78 | Rot. torticollis |
| Laterocollis | 0.308 | **0.279** | **−17.73** | Laterocollis |

## Files
| File | Contents |
|---|---|
| `dystonia_pose_features.py` | landmarks, features, rule, plots |
| `dystonia_pose_features.ipynb` | driver notebook |
| `report.pdf` | full report |
| `figures/` | feature diagrams, overlays, comparison |

## Data
Any frontal upper-body photos; a reference image of the postures is in `figures/dystonia_types_reference.png`.

## Run
```bash
pip install mediapipe==0.10.14 opencv-python numpy pandas matplotlib
python dystonia_pose_features.py man1.jpg:Normal woman_lat.jpg:Laterocollis ...
```

The full paper-style write-up — method, diagrams, results and discussion — is in [`report.pdf`](report.pdf).
