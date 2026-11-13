### Introducing collective motion models to MOT

We intend to use an existing Multiple-Object Tracking like OC-Sort (or ByteTrack) that utilizes Kalman filters for predictions and try to improve on that design using collective motion models. We expect that this could boost the original tracker's performance by a few percentage points on animal datasets like 3D-ZeF or SweetFish.

### Code

#### Setup
```bash
conda env create -f environment.yml      # creates the "mot" env and installs this package (editable)
conda activate mot
```
Datasets are read from `~/Desktop/Data/<dataset>` (override with `CMOT_DATA_ROOT`).

#### Layout
- `src/cmot/trackers/<tracker>/` - one self-contained folder per tracker; `ocsort/motion/` holds OC-SORT's motion models (the focus of the experiments)
- `src/cmot/datasets/` - SweetFish, 3D-ZeF (per camera view), AnimalTrack readers
- `src/cmot/detection/` - YOLO training / export, cached detections, or `gt_oracle` (noisy GT boxes)
- `src/cmot/evaluation/` - TrackEval (HOTA / MOTA / IDF1)
- `configs/` - datasets, detectors, trackers, motion models, experiments
- `weights/` (gitignored) - `<model>/<dataset>/<run_id>/weights/best.pt`
- `outputs/` (gitignored) - cached detections, `runs/<dataset>/<split>/<tracker>/<motion>/<run_id>/` plus `index.csv`; nothing is ever overwritten

#### Run
```bash
python scripts/train_detector.py configs/detectors/yolo11s_sweetfish.yaml
python scripts/export_detections.py weights/yolo11s/sweetfish/<run_id> --splits val_half
python scripts/run_tracking.py configs/experiments/ocsort_sweetfish_yolo.yaml detections.det_id=<det_id>
python scripts/run_tracking.py configs/experiments/ocsort_sweetfish_oracle.yaml tracker.overrides.inertia=0.1
python scripts/evaluate.py outputs/runs/sweetfish/val_half/ocsort/kf_cv_xysr/<run_id>   # re-evaluate an existing run
```

### Starting point 
- download the datasets
- implement one of the proposed trackers
- create an evaluation environment


### Milestones
#### 1st milestone
- data preparation
- baseline implementation and evaluation
- report available [here](Report.pdf) (LaTeX source in [report/](report/Research_mathematics.tex))

What we did:
- **Literature review** - tracking-by-detection and SORT-style trackers (SORT, DeepSORT, ByteTrack, OC-SORT), segmentation-based tracking (SAM 2/3), alternative motion models (constant acceleration, turn rate) and models of collective behaviour (Aoki, Reynolds' boids, Couzin; fitted to real fish by Katz et al. and Heras et al.). We found that collective motion models have so far not been used as the motion model of SORT-style trackers.
- **Data** - readers for SweetFish, 3D-ZeF and AnimalTrack. Findings and decisions:
  - SweetFish's actual resolution / FPS (2456x2058, 30 FPS) differ from what the original paper states.
  - The SweetFish GT file (`train_gt_mot.txt`) stores x and y swapped. We verified this against fish positions in the video (median error 68 px swapped vs 587 px as given) and fix it on load (`swap_xy: true`).
  - Only the SweetFish train video (10,000 frames, 10 fish, fixed 100x100 boxes) is annotated, so we split it in time MOT17-style: `train_half` (frames 1-5000) and `val_half` (5001-10000). The detector trains on frames 1-4500 and uses 4501-5000 for checkpoint selection, so tracking on `val_half` only sees unseen frames.
  - 3D-ZeF uses the official train / val / test splits (the test fish are a different, younger cohort). Each sequence is tracked per camera view as a separate 2D sequence (`-T` top, `-F` front).
  - The MOT17 pedestrian output filters (min box area, max aspect ratio) are disabled for the fish datasets.
- **Baseline** - OC-SORT reimplemented as a self-contained tracker (OCM, OCR, ORU, optional BYTE) with defaults from the official repository. It is a line-by-line port of the official `OCSort.update`, and with the default config its outputs match the official code. Its constant-velocity Kalman filter (`kf_cv_xysr`) and the ORU virtual trajectory sit behind pluggable interfaces (`MotionModel` / `InteractionModel`, `TRAJECTORIES`), so a collective motion model can be dropped in without touching association.
- **Detections** - YOLO11s training and detection export/caching for SweetFish and 3D-ZeF, plus a `gt_oracle` source (GT boxes with position/size noise and dropped detections) to evaluate the tracker independently of detector quality. The 3D-ZeF detector trains on the official train split and validates on val. Detections are cached per weights (`det_id`), so all motion-model variants see the same detections and any differences between runs come from tracking alone.
- **Evaluation** - TrackEval integration (HOTA, DetA, AssA, IDF1, IDSW, MOTA). Every run is stored with its resolved config, git info and metrics and logged to `index.csv`.
- **Method (report)** - formulated the collective motion model: each track sees neighbours within a radius `r_v` and field of view `phi`; cohesion, separation and alignment drives are weighted into an acceleration that enters the Kalman predict step as a control input `B a_i`. Without neighbours it reduces to the original constant-velocity model, and the covariance and update steps are unchanged. All accelerations are computed from the `t-1` estimates before any track is predicted, and the ORU replay runs without the control input.
- **Report** - introduction, methods (motion model, OC-SORT association) and experimental setup (datasets, metrics) written.

#### 2nd milestone
- implementation and evaluation of different collective motion models

#### 3rd milestone
- additional collective motion experiments
- adding newer baselines to compare against current SOTA (SAM3.1 & DAM4SAM for example)
- aggregating the results and writting the final report


### Notes
- use an existing MOT tracker using Kalman filters like OC-Sort or ByteTrack
- look at how collective motion in MOT was already done
- test whether other Kalman motion models already improve performance -- by default constant velocity, we can try constant acceleration for example
- swap out Kalman for collective motion models
- evaluate on AnimalTrack/3D-ZeF/SweetFish
- AnimalTrack has the problem of camera motion which is not an area of focus for this work so it would be better to just focus on SweetFish and 3D-ZeF

### Github identities
- jp7316 -> Jure Pahor (63210230)
- InternatBlackhole -> Timotej Kroflič (63210171)
- jimocool2 -> Jure Dežman (63210223)

### References 
#### Prior work in domain
- You’ll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking [https://ieeexplore.ieee.org/document/5459260]
- Who are you with and where are you going? [https://ieeexplore.ieee.org/document/5995468]
- MotionTrack: Learning Robust Short-term and Long-term Motions
for Multi-Object Tracking [https://arxiv.org/abs/2303.10404]

#### Potential baselines
- Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking
[https://arxiv.org/abs/2203.14360]
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box [https://arxiv.org/abs/2110.06864]

#### Potential datasets
- AnimalTrack: A Benchmark for Multi-Animal Tracking in the Wild [https://arxiv.org/abs/2205.00158]
- Fish Tracking Challenge 2024: A Multi-Object Tracking Competition with Sweetfish Schooling Data [https://arxiv.org/abs/2409.00339]
- 3D-ZeF: A 3D Zebrafish Tracking Benchmark Dataset [https://arxiv.org/abs/2006.08466]
