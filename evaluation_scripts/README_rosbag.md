# DROID-SLAM Rosbag Benchmark

Run DROID-SLAM directly on ROS bag files — no image export, no live ROS nodes needed.
Supports TSRB (RealSense D435i, compressed) and Gazebo simulation datasets with RGB-D depth.

## Prerequisites

All dependencies are installed inside the DROID-SLAM venv.

```bash
cd /path/to/DROID-SLAM
source .venv/bin/activate

# rosbags: pure-Python ROS1 bag reader (installed as an evo dependency)
pip show rosbags   # should show version 0.9.x
```

No ROS installation or `source /opt/ros/noetic/setup.bash` needed.

### DROID-SLAM weights

```bash
./tools/download_model.sh   # places droid.pth in the DROID-SLAM root
```

---

## Dataset layout

**TSRB** (`/mnt/IVALAB/rosbags/tsrb/multi_run/`):
```
multi_run/
├── path1_1_ordered/
│   └── path1_1_0.bag          # one bag per sequence
├── path1_2_ordered/
│   └── ...
└── gt_poses/
```

**Gazebo** (`/mnt/IVALAB/rosbags/gazebo/multi_run/`):
```
multi_run/
├── seq_metadata.csv
├── aws_hospital/
│   ├── disordered/
│   │   ├── path10x3/
│   │   │   └── _0.bag         # single segment
│   │   ├── path10x5/
│   │   │   ├── _0.bag         # multi-segment: played as one continuous run
│   │   │   └── _1.bag
│   │   └── ...
│   └── ordered/
│       ├── _0.bag             # three segments of one ordered run
│       ├── _1.bag
│       └── _2.bag
└── aws_small_house/
    └── ...
```

> **Multi-segment bags**: when a directory contains `_0.bag`, `_1.bag`, … these are
> segments of *one* continuous recording split for size control. They are played
> sequentially as a single DROID-SLAM session — not as independent runs.

---

## Topics and depth

| Dataset | Color topic | Depth topic | Depth alignment |
|---------|-------------|-------------|-----------------|
| tsrb | `/camera/color/image_raw/compressed` | `/camera/aligned_depth_to_color/image_raw/compressedDepth` | none (pre-aligned) |
| gazebo | `/camera/color/image_raw` | `/camera/depth/image_raw` | projected to color frame (different focal lengths) |

Camera intrinsics are read automatically from each bag's `/camera/*/camera_info` topic.

---

## Running commands

All commands run from the DROID-SLAM root with the venv active:

```bash
cd /path/to/DROID-SLAM
source .venv/bin/activate
```

### Single TSRB sequence

```bash
python evaluation_scripts/bench_rosbag.py \
    --dataset tsrb \
    --seq path1_1_ordered \
    --result_dir /mnt/DATA/experiments/tmp/tsrb/droid_slam/
```

### Single Gazebo sequence

```bash
python evaluation_scripts/bench_rosbag.py \
    --dataset gazebo \
    --seq aws_hospital_path10x3 \
    --result_dir /mnt/DATA/experiments/tmp/gazebo/droid_slam/
```

Multi-segment sequences (e.g. `aws_hospital_path10x5` with `_0.bag` + `_1.bag`) are
handled automatically — all segments are played as one continuous session.

### Full TSRB benchmark (all sequences)

```bash
python evaluation_scripts/bench_rosbag.py \
    --dataset tsrb \
    --result_dir /mnt/DATA/experiments/droid_slam/ \
    --num_rounds 3
```

### Full Gazebo benchmark (all sequences in seq_metadata.csv)

```bash
python evaluation_scripts/bench_rosbag.py \
    --dataset gazebo \
    --result_dir /mnt/DATA/experiments/droid_slam/ \
    --num_rounds 3
```

### Quick test — single bag file (bypasses sequence discovery)

```bash
# TSRB
python evaluation_scripts/bench_rosbag.py \
    --bagfile /mnt/IVALAB/rosbags/tsrb/multi_run/path1_1_ordered/path1_1_0.bag \
    --dataset tsrb --seq path1_1_ordered \
    --result_dir /tmp/droid_test/

# Gazebo (single-segment sequence)
python evaluation_scripts/bench_rosbag.py \
    --bagfile /mnt/IVALAB/rosbags/gazebo/multi_run/aws_hospital/disordered/path10x3/_0.bag \
    --dataset gazebo --seq aws_hospital_path10x3 \
    --result_dir /tmp/droid_test/
```

> **Note**: `--bagfile` only accepts one path. For multi-segment sequences use
> `--dataset` + `--seq` so all segments are discovered and played together.

---

## Output files (per sequence, per round)

Saved under `{result_dir}/{dataset}/droid_slam/Round{N}/`:

| File | Contents |
|------|----------|
| `{seq}_AllFrameTrajectory.txt` | All frames — keyframes + interpolated (TUM format) |
| `{seq}_KeyframeTrajectory.txt` | Keyframes only, after global BA (TUM format) |
| `{seq}_latency.txt` | Per-frame `image_timestamp  elapsed_seconds` |

**TUM format:** `timestamp tx ty tz qx qy qz qw` (space-separated, one pose per line)

---

## Key parameters

| Flag | Default | Notes |
|------|---------|-------|
| `--stride` | 2 | Frame stride — 1 = every frame, 2 = every other |
| `--image_size H W` | 240 320 | Resize target. 240 320 ≈ 2.9 GB GPU; 384 512 ≈ 7.3 GB |
| `--filter_thresh` | 3.5 | Min optical flow (px) to accept a new keyframe |
| `--keyframe_thresh` | 4.0 | Min distance (px) between keyframes |
| `--frontend_window` | 25 | Local BA sliding window size |
| `--backend_radius` | 2 | Edge radius for global BA graph (higher = denser) |
| `--backend_thresh` | 22.0 | Flow threshold for global BA edges |
| `--buffer` | 2048 | Keyframe buffer size (GPU memory ∝ buffer × H × W) |
| `--num_rounds` | 1 | Repeat each sequence N times |
| `--vis` | off | Enable 3D point-cloud viewer |

### Recommended settings for indoor robot (12 GB GPU)

```bash
--stride 1 --filter_thresh 2.5 --backend_radius 3 --backend_thresh 18.0
```

For higher accuracy (needs 16+ GB GPU):

```bash
--image_size 384 512 --stride 1 --filter_thresh 2.5 --backend_radius 3
```

---

## Architecture notes

DROID-SLAM has no loop-closure detector. It uses a three-stage pipeline:

1. **Motion filter** — selects keyframes (frames with `> filter_thresh` pixel flow)
2. **Frontend (local BA)** — sliding window of `frontend_window` keyframes, runs every frame
3. **Backend (global BA)** — `terminate()` only: 7 + 12 iterations over *all* keyframes

`KeyframeTrajectory.txt` contains poses after stage 3 (global refinement).
`AllFrameTrajectory.txt` additionally fills non-keyframe poses by interpolation.

Depth is fed to every `droid.track()` call, anchoring absolute scale and populating
`DepthVideo.disps_sens` so the backend skips monocular `normalize()`.

### Typical tracking latency

| Dataset | Mean latency | Reason |
|---------|-------------|--------|
| TSRB | ~7 ms | 30 Hz camera → small inter-frame motion → few keyframes → mostly filter-only |
| Gazebo | ~35 ms | ~11 Hz camera → larger motion per frame → more keyframes → more BA calls |

---

## Troubleshooting

**`Topic '...' not found in bag`**
Inspect topics:
```bash
python -c "
from rosbags.rosbag1 import Reader
r = Reader('file.bag'); r.open()
print([c.topic for c in r.connections])
r.close()
"
```

**GPU OOM**
Reduce `--image_size` (try `240 320`) or lower `--buffer` (try `1024`).

**Large drift**
DROID-SLAM has no loop closure. Mitigations:
- Lower `--stride` (more constraints per unit time)
- Lower `--filter_thresh` (more keyframes)
- Raise `--backend_radius` and lower `--backend_thresh` (denser global BA graph)
- Use higher `--image_size` for better feature quality
