### Introducing collective motion models to MOT

We intend to use an existing Multiple-Object Tracking like OC-Sort or ByteTrack that utilizes Kalman filters for predictions and try to improve on that design using collective motion models. We expect that this could boost the original tracker's performance by a few percentage points on animal datasets like 3D-ZeF or SweetFish.

### Starting point 
- download the datasets
- implement one of the proposed trackers
- create an evaluation environment


### Milestones
#### 1st milestone
- data preparation
- baseline implementation and evaluation

#### 2nd milestone
- implementation and evaluation of different collective motion models

#### 3rd milestone
- additional collective motion experiments
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
