"""
DROID-SLAM benchmark on TSRB / Gazebo rosbags.

Reads bag files directly — no image export, no live ROS nodes needed.
Intrinsics are read from the bag's CameraInfo topic for accuracy.
Output trajectories are saved in TUM format: timestamp tx ty tz qx qy qz qw

Usage (from the DROID-SLAM root with .venv active):

  # Full dataset run
  python evaluation_scripts/bench_rosbag.py --dataset tsrb \\
      --result_dir /mnt/DATA/experiments/droid_slam/

  python evaluation_scripts/bench_rosbag.py --dataset gazebo \\
      --result_dir /mnt/DATA/experiments/droid_slam/

  # Single bag (for quick testing)
  python evaluation_scripts/bench_rosbag.py \\
      --bagfile /mnt/IVALAB/rosbags/tsrb/multi_run/path1_1_ordered/path1_1_0.bag \\
      --dataset tsrb --seq path1_1_ordered \\
      --result_dir /tmp/droid_test/

Note: rosbag must be importable. Either source /opt/ros/noetic/setup.bash
before activating the venv, or rosbag_stream.py adds the path automatically.
"""

import sys
import os
import copy
import time
import argparse
import glob
import traceback
from pathlib import Path

import numpy as np
import torch
import lietorch
from tqdm import tqdm

sys.path.append("droid_slam")
sys.path.append(str(Path(__file__).parent))

from droid import Droid
from droid_async import DroidAsync
from rosbag_stream import RosbagRGBDStream, read_camera_info


# ── Dataset configuration ──────────────────────────────────────────────────────

BAG_ROOTS = {
    "tsrb": "/mnt/IVALAB/rosbags/tsrb/multi_run",
    "gazebo": "/mnt/IVALAB/rosbags/gazebo/multi_run",
}

# Color image topics per dataset
COLOR_TOPICS = {
    "tsrb": "/camera/color/image_raw/compressed",    # CompressedImage JPEG
    "gazebo": "/camera/color/image_raw",              # raw sensor_msgs/Image
}

# Depth topics — TSRB already aligned; Gazebo needs projection alignment
DEPTH_TOPICS = {
    "tsrb":   "/camera/aligned_depth_to_color/image_raw/compressedDepth",
    "gazebo": "/camera/depth/image_raw",
}

# Camera info topics for color (used for tracking intrinsics)
CAMERA_INFO_TOPICS = {
    "tsrb": "/camera/color/camera_info",
    "gazebo": "/camera/color/camera_info",
}

# Camera info topics for depth (needed only for Gazebo alignment)
DEPTH_INFO_TOPICS = {
    "tsrb":   "/camera/aligned_depth_to_color/camera_info",  # same as color; not needed
    "gazebo": "/camera/depth/camera_info",                    # different focal length
}

# Gazebo depth camera has different intrinsics → align to color frame before use
DEPTH_NEEDS_ALIGNMENT = {
    "tsrb":   False,   # aligned_depth_to_color already in color frame
    "gazebo": True,    # depth camera has fx=347.998 vs color fx=462.138
}

# Fallback intrinsics [fx, fy, cx, cy] @ 640x480 — from actual bag CameraInfo
FALLBACK_INTRINSICS = {
    "tsrb":   [610.7184, 610.9886, 321.7739, 243.8619],  # RealSense D435i
    "gazebo": [462.1380, 462.1380, 320.0,    240.0],      # Gazebo virtual camera
}

TSRB_SEQUENCES = [
    "path1_1_ordered",
    "path1_1_1_ordered",
    "path1_2_ordered",
    "path2_1_ordered",
    "north_path_1_disordered",
    "path1_1_disordered",
    "path1_2_disordered",
    "path1_3_disordered",
]


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _gazebo_subpath(seq):
    """
    Map a Gazebo sequence name to its subdirectory under BAG_ROOTS['gazebo'].

    All sequences in seq_metadata.csv live under {env}/disordered/{path}/.
    The ordered bags at {env}/ordered/ have no named path subdirectory —
    they are referenced directly by path (e.g. aws_hospital/ordered).

    Search order: disordered/path → ordered/path → ordered/ (multi-bag)
    """
    root = Path(BAG_ROOTS["gazebo"])
    parts = seq.split("_path", 1)
    if len(parts) != 2:
        return seq
    env, rest = parts
    path_name = f"path{rest}"

    for candidate in [
        Path(env) / "disordered" / path_name,
        Path(env) / "ordered"    / path_name,
        Path(env) / "ordered",
    ]:
        if (root / candidate).is_dir() and list((root / candidate).glob("*.bag")):
            return str(candidate)

    return str(Path(env) / "disordered" / path_name)  # fallback (will raise on missing)


def find_bagfiles(dataset, seq):
    """Return all bag segments for a sequence, sorted chronologically."""
    root = Path(BAG_ROOTS[dataset])
    if dataset == "gazebo":
        pattern = str(root / _gazebo_subpath(seq) / "*.bag")
    else:
        pattern = str(root / seq / "*.bag")
    bags = sorted(glob.glob(pattern))
    if not bags:
        raise FileNotFoundError(f"No bag found for {dataset}/{seq} (pattern: {pattern})")
    return bags


def get_sequences(dataset):
    if dataset == "tsrb":
        return TSRB_SEQUENCES
    elif dataset == "gazebo":
        csv = Path(BAG_ROOTS["gazebo"]) / "seq_metadata.csv"
        if csv.is_file():
            import pandas as pd
            return pd.read_csv(csv)["sequence"].tolist()
        # fallback: scan directories
        seqs = []
        root = Path(BAG_ROOTS["gazebo"])
        for env_dir in sorted(root.iterdir()):
            if not env_dir.is_dir():
                continue
            for sub_dir in sorted(env_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue
                for bag in sorted(sub_dir.glob("*.bag")):
                    seqs.append(f"{env_dir.name}_{bag.stem}")
        return seqs
    return []


# ── Trajectory / latency I/O ───────────────────────────────────────────────────

def save_tum_trajectory(traj_est, timestamps, path):
    """
    Save in TUM format: timestamp tx ty tz qx qy qz qw

    DepthVideo stores poses as [tx, ty, tz, qx, qy, qz, qw] (xyzw quaternion).
    terminate() returns camera_trajectory.inv().data, same layout → already TUM.
    """
    n = min(len(timestamps), len(traj_est))
    ts = np.array(timestamps[:n], dtype=np.float64).reshape(-1, 1)
    np.savetxt(path, np.concatenate([ts, traj_est[:n]], axis=1), fmt="%.9f")


def save_latency(records, path):
    """
    Save per-frame tracking latency.
    records: list of (image_timestamp, elapsed_seconds)
    Columns: image_timestamp  elapsed_seconds
    """
    arr = np.array(records, dtype=np.float64)
    np.savetxt(path, arr, fmt="%.9f", header="image_timestamp elapsed_seconds")


def extract_keyframe_trajectory(droid):
    """
    Extract keyframe-only poses from the video buffer after terminate().

    These are the frames selected by the motion filter and refined by both
    frontend (local BA) and backend (global BA). The backend in terminate()
    runs 7+12 iterations of global BA over all keyframes — equivalent to
    loop-closure correction + global refinement.

    Returns (timestamps [K], poses [K,7]) or (None, None) if empty.
    Poses are inverted (same as terminate()) to give world-frame positions.
    """
    video = droid.video2 if hasattr(droid, "video2") else droid.video
    k = video.counter.value
    if k == 0:
        return None, None
    tstamps = video.tstamp[:k].cpu().numpy()
    # video.poses stores c_T_w; invert → w_T_c (world frame), same as terminate()
    poses_inv = lietorch.SE3(video.poses[:k].clone()).inv().data.cpu().numpy()
    return tstamps, poses_inv


# ── Core sequence runner ───────────────────────────────────────────────────────

def run_sequence(bagfiles, dataset, seq, round_dir, args):
    """
    Run DROID-SLAM on one sequence (one or more bag segments) and save outputs.

    bagfiles may be a single path string or a list of paths that form one
    continuous recording split across multiple files for size control.
    """
    if isinstance(bagfiles, str):
        bagfiles = [bagfiles]

    round_dir = Path(round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    traj_file    = round_dir / f"{seq}_AllFrameTrajectory.txt"
    kf_traj_file = round_dir / f"{seq}_KeyframeTrajectory.txt"
    latency_file = round_dir / f"{seq}_latency.txt"

    first_bag = bagfiles[0]

    # Read camera intrinsics from the first bag segment (same for all segments)
    color_K = read_camera_info(first_bag, CAMERA_INFO_TOPICS[dataset])
    if color_K is None:
        print("  [WARN] CameraInfo not in bag — using fallback intrinsics.")
        color_K = FALLBACK_INTRINSICS[dataset]

    # Read depth camera intrinsics when alignment is required (Gazebo)
    depth_K = None
    if DEPTH_NEEDS_ALIGNMENT[dataset]:
        depth_K = read_camera_info(first_bag, DEPTH_INFO_TOPICS[dataset])
        if depth_K is None:
            print("  [WARN] Depth CameraInfo not found; alignment will be skipped.")

    if len(bagfiles) > 1:
        print(f"  {len(bagfiles)} bag segments — played as one continuous sequence")

    # Build RGB-D stream across all bag segments
    stream = RosbagRGBDStream(
        bagfile=bagfiles,
        color_topic=COLOR_TOPICS[dataset],
        depth_topic=DEPTH_TOPICS[dataset],
        color_K=color_K,
        depth_K=depth_K,
        align_depth=DEPTH_NEEDS_ALIGNMENT[dataset],
        target_area=tuple(args.image_size),
        stride=args.stride,
    )
    # Color-only view for droid.terminate() (second bag pass, 3-tuple API)
    color_stream = stream.as_color_stream()

    # Approximate frame count from bag metadata (no messages read yet)
    n_approx = len(stream)
    img_size = stream.image_size()   # reads only the first frame
    bag_label = ", ".join(Path(b).name for b in bagfiles)
    print(f"  ~{n_approx} frames from [{bag_label}]  (image_size={img_size})")

    # Use a per-run copy of args so image_size doesn't bleed across sequences
    run_args = copy.copy(args)
    run_args.image_size = img_size

    droid = DroidAsync(run_args) if run_args.asynchronous else Droid(run_args)
    latency_records = []   # (timestamp, elapsed_s) — collected during tracking
    n_depth_missing = 0
    try:
        # First pass: feed RGB-D frames to DROID-SLAM one at a time
        for tstamp, image, depth, intrinsics in tqdm(stream, total=n_approx, desc=seq):
            t0 = time.perf_counter()
            droid.track(tstamp, image, depth=depth, intrinsics=intrinsics)
            latency_records.append((tstamp, time.perf_counter() - t0))
            if depth is None:
                n_depth_missing += 1

        if n_depth_missing:
            print(f"  [WARN] {n_depth_missing}/{len(latency_records)} frames had no depth match")

        # Second pass: terminate() re-reads color frames to fill non-keyframe poses
        traj_est = droid.terminate(color_stream)

        # Extract keyframes BEFORE deleting droid (video buffer still live)
        kf_tstamps, kf_poses = extract_keyframe_trajectory(droid)
    finally:
        del droid
        torch.cuda.empty_cache()

    # Timestamps already collected; no third bag pass needed
    timestamps = [r[0] for r in latency_records]
    save_tum_trajectory(traj_est, timestamps, traj_file)
    save_latency(latency_records, latency_file)
    if kf_tstamps is not None:
        save_tum_trajectory(kf_poses, kf_tstamps.tolist(), kf_traj_file)
        print(f"  Keyframes: {len(kf_tstamps)} / {n_approx} frames")

    mean_ms = np.mean([r[1] for r in latency_records]) * 1000
    print(f"  Mean track latency: {mean_ms:.1f} ms  ({1000/mean_ms:.1f} fps)")
    print(f"  Saved → {traj_file}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run DROID-SLAM on TSRB / Gazebo rosbags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / sequence selection
    p.add_argument("--dataset", default="tsrb", choices=["tsrb", "gazebo"])
    p.add_argument("--bagfile", default=None,
                   help="Single bag path — skips dataset discovery, runs one bag only")
    p.add_argument("--seq",     default=None,
                   help="Sequence name: used as output label with --bagfile, "
                        "or as a filter to run only this sequence in dataset mode")
    p.add_argument("--result_dir", default="/mnt/DATA/experiments/droid_slam/")
    p.add_argument("--num_rounds", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")

    # Feed control
    p.add_argument("--stride", type=int, default=2,
                   help="Frame stride (2 = every other frame, halving the rate)")

    # DROID-SLAM core
    p.add_argument("--weights", default="droid.pth")
    p.add_argument("--buffer",  type=int, default=2048,
                   help="Keyframe buffer size (GPU memory ∝ buffer × H × W)")
    p.add_argument("--image_size", type=int, nargs=2, default=[240, 320],
                   metavar=("H", "W"),
                   help="Resize images to this resolution before feeding DROID. "
                        "240 320 (~2.9 GB for buffer=2048) fits 12 GB GPU. "
                        "384 512 (~7.3 GB for buffer=2048) needs 16+ GB GPU.")
    p.add_argument("--vis", action="store_true",
                   help="Enable 3D visualization (requires: pip install moderngl moderngl-window)")
    p.add_argument("--upsample",    action="store_true")
    p.add_argument("--asynchronous", action="store_true")
    p.add_argument("--frontend_device", default="cuda")
    p.add_argument("--backend_device",  default="cuda")

    # Motion / keyframe thresholds (tuned for indoor 15–30 fps robot data)
    p.add_argument("--beta",             type=float, default=0.3)
    p.add_argument("--filter_thresh",    type=float, default=3.5)
    p.add_argument("--warmup",           type=int,   default=8)
    p.add_argument("--keyframe_thresh",  type=float, default=4.0)

    # Frontend (local BA)
    p.add_argument("--frontend_thresh",  type=float, default=16.0)
    p.add_argument("--frontend_window",  type=int,   default=25)
    p.add_argument("--frontend_radius",  type=int,   default=2)
    p.add_argument("--frontend_nms",     type=int,   default=1)

    # Backend (global BA)
    p.add_argument("--backend_thresh",   type=float, default=22.0)
    p.add_argument("--backend_radius",   type=int,   default=2)
    p.add_argument("--backend_nms",      type=int,   default=3)

    p.add_argument("--motion_damping",   type=float, default=0.5)

    args = p.parse_args()
    args.stereo = False
    args.disable_vis = not args.vis
    return args


def main():
    torch.multiprocessing.set_start_method("spawn")
    args = parse_args()
    result_dir = Path(args.result_dir)

    # ── Single bag mode ──────────────────────────────────────────────────────
    if args.bagfile is not None:
        seq = args.seq or Path(args.bagfile).stem
        round_dir = result_dir / args.dataset / "droid_slam" / "Round1"
        run_sequence(args.bagfile, args.dataset, seq, round_dir, args)
        return

    # ── Multi-sequence mode ──────────────────────────────────────────────────
    sequences = get_sequences(args.dataset)
    if not sequences:
        print(f"No sequences found for dataset '{args.dataset}'.")
        return

    if args.seq is not None:
        if args.seq not in sequences:
            print(f"Sequence '{args.seq}' not found in {args.dataset}. Available:")
            for s in sequences:
                print(f"  {s}")
            return
        sequences = [args.seq]

    for seq in sequences:
        try:
            bagfiles = find_bagfiles(args.dataset, seq)
        except FileNotFoundError as e:
            print(f"  Skipping {seq}: {e}")
            continue

        if len(bagfiles) > 1:
            print(f"  {seq}: {len(bagfiles)} bag segments (played as one sequence)")

        for round_idx in range(1, args.num_rounds + 1):
            # All bag segments together form one run; each round repeats the full sequence
            print(f"\n=== droid_slam | {args.dataset} | {seq} | Round {round_idx} ===")
            round_dir = result_dir / args.dataset / "droid_slam" / f"Round{round_idx}"
            traj_file = round_dir / f"{seq}_AllFrameTrajectory.txt"

            if not args.overwrite and traj_file.is_file():
                print("  Already done, skipping.")
                continue

            try:
                run_sequence(bagfiles, args.dataset, seq, round_dir, args)
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return
            except Exception as e:
                print(f"  [ERROR] {seq} round {round_idx}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
