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
```

### Starting point 
- download the datasets
- implement one of the proposed trackers
- create an evaluation environment


### Milestones
#### 1st milestone
- data preparation
- baseline implementation and evaluation
- report available [here](Report.pdf)

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
