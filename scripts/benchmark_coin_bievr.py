#!/usr/bin/env python3
"""Replay local paper datasets and measure translation ATE, latency and memory.

Requires numpy and PyYAML. Each install prefix must contain a separately built
bievr_lio_ros2 package; recordings and original ground truth remain untouched.
"""

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml


NCD = "newer_college/2021-ouster-os0-128-alphasense"
SEQUENCES = {
    "tunnel_d": ("enwide", "enwide/tunnel_d/2023-08-08-17-50-31-tunnel_d",
                 "enwide/tunnel_d/gt-tunnel_d.csv"),
    "tunnel_s": ("enwide", "enwide/tunnel_s/2023-08-08-17-12-37-tunnel_s",
                 "enwide/tunnel_s/gt-tunnel_s.csv"),
    "shield1": ("geode_gamma", "geode/Shield_tunnel1_gamma_pc2",
                "geode/Shield_tunnel1_gt.txt"),
    "quad_hard": ("ncd", f"{NCD}/c1_newer_college/2021-07-01-11-35-14_0-quad-hard_ros2",
                  f"{NCD}/c1_newer_college/ground_truth/tum_format/gt-nc-quad-hard.csv"),
}


def offset_positions(trajectory, lever):
    """Position of a body-fixed comparison origin: p_W_C = p_W_I + R_W_I t_I_C."""
    quaternion = trajectory[:, 4:8]
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("estimated poses require finite, nonzero quaternions")
    quaternion = quaternion / norms[:, None]
    xyz, w = quaternion[:, :3], quaternion[:, 3:4]
    rotated = lever + 2 * np.cross(xyz, np.cross(xyz, lever) + w * lever)
    return trajectory[:, 1:4] + rotated


def prism_positions(trajectory):
    """ENWIDE T_IMU_PRISM from prism_imu_extrinsics.txt."""
    return offset_positions(trajectory, np.array([-0.006253, 0.011775, 0.10825]))


def geode_lever():
    """Inverse translation from the official gamma2GT_leica.py device transform.

    https://github.com/PengYu-Team/GEODE_dataset/blob/main/script/gamma2GT_leica.py
    The reference constructs R from the rounded quaternion without normalizing it;
    solve that same matrix to reproduce its T_device @ inverse(T) translation.
    """
    w, x, y, z = .999901, -.00492765, .00575961, .0117651
    rotation = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])
    return -np.linalg.solve(rotation, [.00947221, -.308202, -.365733])


def comparison_positions(trajectory, sensor):
    if sensor == "enwide":
        return prism_positions(trajectory)
    if sensor == "ncd":
        # Multi-camera GT uses Base. The released os_imu_lidar_transforms.yaml
        # gives T_base_imu = T_base_sensor * T_sensor_imu; invert its translation.
        return offset_positions(trajectory, np.array([.013, -.012, -.106]))
    if sensor == "geode_gamma":
        return offset_positions(trajectory, geode_lever())
    return trajectory[:, 1:4]


def aligned_errors(estimate, truth):
    """Rigid least-squares alignment of corresponding positions, without scale."""
    centered_est = estimate - estimate.mean(axis=0)
    centered_gt = truth - truth.mean(axis=0)
    u, _, vt = np.linalg.svd(centered_est.T @ centered_gt)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    aligned = centered_est @ (u @ correction @ vt) + truth.mean(axis=0)
    return np.linalg.norm(aligned - truth, axis=1)


def evaluate(trajectory_path, gt_path, prism=False, *, sensor=None, max_interpolation_gap=0.2):
    estimate = np.atleast_2d(np.loadtxt(trajectory_path))
    truth = np.atleast_2d(np.loadtxt(gt_path))
    if estimate.shape[1] != 8 or not np.isfinite(estimate).all():
        raise ValueError("estimated trajectory must contain finite TUM poses")
    if truth.shape[1] < 4:
        raise ValueError("ground truth requires timestamps and xyz positions")
    truth = truth[np.isfinite(truth[:, :4]).all(axis=1)]
    truth = truth[np.argsort(truth[:, 0])]
    if len(estimate) < 3 or len(truth) < 3:
        raise ValueError("at least three trajectory samples are required")
    stamps = estimate[:, 0]
    if np.any(np.diff(stamps) <= 0):
        raise ValueError("estimated timestamps must be strictly increasing")
    if not np.isfinite(max_interpolation_gap) or max_interpolation_gap <= 0:
        raise ValueError("maximum interpolation gap must be positive and finite")
    sensor = sensor if sensor is not None else ("enwide" if prism else "")
    positions = comparison_positions(estimate, sensor)
    target_stamps = truth[:, 0]
    insertion = np.searchsorted(stamps, target_stamps)
    right = np.clip(insertion, 0, len(stamps) - 1)
    exact = stamps[right] == target_stamps
    left = np.where(exact, right, np.clip(insertion - 1, 0, len(stamps) - 1))
    in_span = (target_stamps >= stamps[0]) & (target_stamps <= stamps[-1])
    intervals = stamps[right] - stamps[left]
    valid = in_span & (exact | ((insertion > 0) & (insertion < len(stamps)) &
                                (intervals <= max_interpolation_gap)))
    if valid.sum() < 3:
        raise ValueError("fewer than three GT timestamps have bounded estimate interpolation support")
    weight = np.zeros(valid.sum())
    np.divide(target_stamps[valid] - stamps[left[valid]], intervals[valid],
              out=weight, where=intervals[valid] > 0)
    interpolated = ((1 - weight[:, None]) * positions[left[valid]] +
                    weight[:, None] * positions[right[valid]])
    errors = aligned_errors(interpolated, truth[valid, 1:4])
    duration = float(stamps[-1] - stamps[0])
    overlap = float(max(0, min(stamps[-1], target_stamps[-1]) -
                        max(stamps[0], target_stamps[0])))
    result = {
        "evaluation_version": 2,
        "association_method": "estimate_interpolated_at_ground_truth",
        "max_interpolation_gap_s": max_interpolation_gap,
        "alignment": "rigid_no_scale",
        "comparison_frame": {"enwide": "prism", "ncd": "base",
                             "geode_gamma": "alpha_device_gt"}.get(sensor, "trajectory_origin"),
        "ate_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ate_max_m": float(errors.max()),
        "matched_samples": int(valid.sum()),
        "trajectory_samples": len(estimate),
        "trajectory_start_s": float(estimate[0, 0]),
        "trajectory_end_s": float(estimate[-1, 0]),
        "trajectory_duration_s": duration,
        "ground_truth_samples": len(truth),
        "ground_truth_samples_in_trajectory": int(in_span.sum()),
        "ground_truth_match_fraction": float(valid.sum() / in_span.sum()),
        "ground_truth_time_overlap_s": overlap,
        "ground_truth_time_coverage": overlap / duration,
        "ground_truth_uncovered_start_s": float(max(0, target_stamps[0] - stamps[0])),
        "ground_truth_uncovered_end_s": float(max(0, stamps[-1] - target_stamps[-1])),
    }
    if sensor == "geode_gamma":
        result["evaluation_caveat"] = (
            "Official gamma2GT_leica.py device-frame correction applied to configured T_W_I. "
            "Existing geode_gamma calibration uses the official external-device extrinsic "
            "with /livox/imu; that input convention is unchanged across compared modes.")
    return result


def timing_metrics(log_path):
    previous_count, previous_sum = 0, 0.0
    latencies = []
    photo_counts = []
    frames = 0
    with log_path.open(errors="replace") as stream:
        for line in stream:
            match = re.match(r"^step\s+(\d+)\s+([\d.]+)", line)
            if match:
                count, total = int(match[1]), float(match[2])
                if count == previous_count + 1:
                    latencies.append(total - previous_sum)
                previous_count, previous_sum = count, total
                frames = max(frames, count)
            match = re.search(r"photometric points: (\d+)", line)
            if match:
                photo_counts.append(int(match[1]))
    result = {"processed_frames": frames}
    if latencies:
        result.update(processing_mean_ms=1000 * float(np.mean(latencies)),
                      processing_p50_ms=1000 * float(np.median(latencies)),
                      processing_p95_ms=1000 * float(np.percentile(latencies, 95)))
    if photo_counts:
        result.update(photometric_points_mean=float(np.mean(photo_counts)),
                      photometric_active_fraction=float(np.mean(np.array(photo_counts) > 0)))
    return result


def add_frame_coverage(result, data, bag, sensor_path):
    metadata_path = data / bag / "metadata.yaml"
    if not metadata_path.exists():
        return
    info = yaml.safe_load(metadata_path.read_text())["rosbag2_bagfile_information"]
    point_topic = yaml.safe_load(sensor_path.read_text())["topics"]["pointcloud"]
    expected_frames = sum(topic["message_count"] for topic in info["topics_with_message_count"]
                          if topic["topic_metadata"]["name"] == point_topic)
    result["expected_frames"] = expected_frames
    result["frame_coverage"] = result.get("processed_frames", 0) / max(1, expected_frames)
    result["complete"] = result.get("exit_code") == 0 and result["frame_coverage"] >= 0.99


def add_evaluation(result, trajectory_path, gt_path, sensor):
    # Remove obsolete evaluation fields while retaining replay timing and memory.
    exact_keys = {"matched_samples", "trajectory_samples", "trajectory_start_s",
                  "trajectory_end_s", "trajectory_duration_s", "association_method",
                  "max_interpolation_gap_s", "alignment", "comparison_frame"}
    for key in list(result):
        if key in exact_keys or key.startswith(("ate_", "ground_truth_", "evaluation_")):
            del result[key]
    try:
        result.update(evaluate(trajectory_path, gt_path, sensor=sensor))
    except (ValueError, OSError, np.linalg.LinAlgError) as error:
        result["evaluation_error"] = str(error)


def evaluate_recording(args, name):
    """Re-evaluate saved trajectories without starting a process or replacing logs."""
    repo = Path(__file__).resolve().parents[1]
    sensor, bag, gt = SEQUENCES[name]
    output = args.output / args.mode / name
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    log_path = output / "replay.log"
    if log_path.exists():
        for key, value in timing_metrics(log_path).items():
            result.setdefault(key, value)
    sensor_path = repo / "config/sensor_configs" / f"{sensor}.yaml"
    add_frame_coverage(result, args.data, bag, sensor_path)
    add_evaluation(result, output / "trajectory.tum", args.data / gt, sensor)
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def run_sequence(args, name):
    repo = Path(__file__).resolve().parents[1]
    sensor, bag, gt = SEQUENCES[name]
    output = args.output / args.mode / name
    output.mkdir(parents=True, exist_ok=True)
    params = yaml.safe_load((repo / "config/params.yaml").read_text())
    params["intensity"]["enabled"] = args.mode == "coin"
    if args.photo_scale is not None:
        params["intensity"]["photo_scale"] = args.photo_scale
    params["optimization"]["lm_debug_print"] = args.solver_debug
    params["max_num_threads"] = args.threads
    params["debug"].update(dashboard=False, log=False, timing=True,
                           trajectory_path=str(output / "trajectory.tum"), map_path="",
                           publish_all_clouds=False)
    params_file = output / "params.yaml"
    params_file.write_text(yaml.safe_dump(params, sort_keys=False))
    sensor_path = repo / "config/sensor_configs" / f"{sensor}.yaml"
    binary = args.install / "bievr_lio_ros2/lib/bievr_lio_ros2/process_bag"
    command = ["/usr/bin/time", "-v", "-o", str(output / "resources.txt"), str(binary),
               "--params_file", str(params_file), "--sensor_config_file", str(sensor_path),
               "--bag", str(args.data / bag)]
    setup = f"source {shlex.quote(str(args.ros_setup))} && "
    setup += f"source {shlex.quote(str(args.install / 'local_setup.bash'))} && "
    setup += shlex.join(command)
    start = time.monotonic()
    print(f"{args.mode}/{name}: replay started", flush=True)
    with (output / "replay.log").open("w") as log:
        process = subprocess.run(["bash", "-c", setup], stdout=log, stderr=subprocess.STDOUT)
    result = {"sequence": name, "mode": args.mode, "exit_code": process.returncode,
              "wall_seconds": time.monotonic() - start, "threads": args.threads,
              "install": str(args.install), "bag": str(args.data / bag)}
    result.update(timing_metrics(output / "replay.log"))
    add_frame_coverage(result, args.data, bag, sensor_path)
    resources = output / "resources.txt"
    if resources.exists():
        match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", resources.read_text())
        if match:
            result["peak_rss_mib"] = int(match[1]) / 1024
    if process.returncode == 0:
        add_evaluation(result, output / "trajectory.tum", args.data / gt, sensor)
        if "evaluation_error" not in result:
            result["realtime_factor"] = result["trajectory_duration_s"] / result["wall_seconds"]
            result["throughput_hz"] = result["processed_frames"] / result["wall_seconds"]
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path)
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Recompute saved result.json evaluation without replay; keeps runtime fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path.home() / "data/data")
    parser.add_argument("--ros-setup", type=Path, default=Path("/opt/ros/kilted/setup.bash"))
    parser.add_argument("--mode", choices=["baseline", "geometry", "coin"], required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--photo-scale", type=float,
                        help="override the residual scale for a controlled sensitivity run")
    parser.add_argument("--solver-debug", action="store_true",
                        help="log per-trial support, cost and weak-direction information")
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=list(SEQUENCES))
    args = parser.parse_args()
    if not args.evaluate_only and args.install is None:
        parser.error("--install is required when replaying")
    args.install = args.install.resolve() if args.install is not None else None
    args.output, args.data = args.output.resolve(), args.data.resolve()
    action = evaluate_recording if args.evaluate_only else run_sequence
    results = [action(args, name) for name in args.sequences]
    return int(any(not r.get("complete", r.get("exit_code") == 0) or
                   "evaluation_error" in r for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
