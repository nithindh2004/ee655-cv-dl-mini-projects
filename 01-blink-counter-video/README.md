# Per-Eye Blink Counter for Patient Video

An end-to-end tool that counts a patient's blinks separately for each eye in a video: a pre-trained detector crops the eyes, a CNN trained on 41,649 eye images classifies open/closed, and a state machine turns frame predictions into blinks.

## Approach
Eye crops from Haar cascades (or MediaPipe FaceMesh with `--detector mediapipe`) → 618k-parameter CNN (≈ 99.8 % val accuracy) → per-eye state machine (blink = ≥ 2 consecutive closed frames).

## Result
| | Left eye (image left) | Right eye (image right) |
|---|---|---|
| Blinks | 0 | 15 |
| Rate | 0 / min | 21.8 / min |

The 0-vs-15 asymmetry should be confirmed on the annotated video (see report, Sec. 4.5).

## Files
| File | Contents |
|---|---|
| `blink_counter.py` | classifier training, eye cropping, blink FSM, plots |
| `blink_counter.ipynb` | driver notebook |
| `report.pdf` | full report |
| `figures/` | pipeline, classifier, timeline, summary |

## Data
Blink Detection Dataset and the patient video `patient.mp4` (not redistributed here).

## Run
```bash
pip install torch torchvision opencv-python mediapipe==0.10.14 pillow matplotlib
python blink_counter.py --data "Blink Detection Dataset" --video patient.mp4 --detector haar
```

The full paper-style write-up — method, diagrams, results and discussion — is in [`report.pdf`](report.pdf).
