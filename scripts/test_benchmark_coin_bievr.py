#!/usr/bin/env python3
"""Numerical and recorded-result regression tests for the benchmark evaluator."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import benchmark_coin_bievr as benchmark


class EvaluationTests(unittest.TestCase):
    def test_rigid_alignment_does_not_fit_scale(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        truth = points @ rotation + [4., -3., 2.]
        self.assertLess(benchmark.aligned_errors(points, truth).max(), 1e-12)
        self.assertGreater(benchmark.aligned_errors(2 * points, truth).mean(), 0.5)

    def test_lever_arm_rotates_with_body_pose(self):
        trajectory = np.array([[0., 10., 20., 30., 0., 0., np.sqrt(.5), np.sqrt(.5)]])
        position = benchmark.offset_positions(trajectory, np.array([1., 0., 0.]))
        np.testing.assert_allclose(position, [[10., 21., 30.]], atol=1e-12)

    def test_interpolates_estimate_at_gt_without_extrapolating_or_bridging_gaps(self):
        times = np.array([0., .1, .2, 1., 1.1, 1.2])
        positions = np.column_stack([times, times**2, times**3])
        truth_times = np.array([-.1, .05, .15, .5, 1.05, 1.15, 2.])
        truth_positions = np.column_stack([np.interp(truth_times, times, positions[:, i])
                                           for i in range(3)])
        estimated = np.column_stack([times, positions, np.zeros((len(times), 3)), np.ones(len(times))])
        truth = np.column_stack([truth_times, truth_positions, np.zeros((len(truth_times), 4))])
        with tempfile.TemporaryDirectory() as directory:
            estimate_path, truth_path = Path(directory) / "estimate.tum", Path(directory) / "truth.tum"
            np.savetxt(estimate_path, estimated)
            np.savetxt(truth_path, truth)
            result = benchmark.evaluate(estimate_path, truth_path)
        self.assertEqual(result["matched_samples"], 4)
        self.assertEqual(result["ground_truth_samples_in_trajectory"], 5)
        self.assertAlmostEqual(result["ground_truth_match_fraction"], 4 / 5)
        self.assertLess(result["ate_rmse_m"], 1e-12)
        self.assertEqual(result["association_method"], "estimate_interpolated_at_ground_truth")

    def test_ncd_base_frame_uses_inverse_sensor_chain(self):
        trajectory = np.array([[0., 0., 0., 0., 0., 0., 0., 1.]])
        np.testing.assert_allclose(benchmark.comparison_positions(trajectory, "ncd"),
                                   [[.013, -.012, -.106]], atol=1e-12)

    def test_geode_offset_matches_official_inverse_transform(self):
        q_w, q_x, q_y, q_z = .999901, -.00492765, .00575961, .0117651
        rotation = np.array([
            [1 - 2*q_y*q_y - 2*q_z*q_z, 2*q_x*q_y - 2*q_z*q_w, 2*q_x*q_z + 2*q_y*q_w],
            [2*q_x*q_y + 2*q_z*q_w, 1 - 2*q_x*q_x - 2*q_z*q_z, 2*q_y*q_z - 2*q_x*q_w],
            [2*q_x*q_z - 2*q_y*q_w, 2*q_y*q_z + 2*q_x*q_w, 1 - 2*q_x*q_x - 2*q_y*q_y],
        ])
        expected = -np.linalg.solve(rotation, [.00947221, -.308202, -.365733])
        trajectory = np.array([[0., 0., 0., 0., 0., 0., 0., 1.]])
        np.testing.assert_allclose(benchmark.comparison_positions(trajectory, "geode_gamma"),
                                   expected[None, :], atol=1e-12)

    def test_evaluate_only_preserves_recorded_runtime_without_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results" / "geometry" / "shield1"
            output.mkdir(parents=True)
            truth_path = root / "data" / benchmark.SEQUENCES["shield1"][2]
            truth_path.parent.mkdir(parents=True)
            times = np.arange(5) * .1
            positions = np.column_stack([times, times**2, times**3])
            estimated = np.column_stack([times, positions, np.zeros((5, 3)), np.ones(5)])
            truth = estimated.copy()
            truth[:, 1:4] = benchmark.comparison_positions(estimated, "geode_gamma")
            truth[:, 4:8] = 0  # Position-only GEODE GT has no valid quaternions.
            np.savetxt(output / "trajectory.tum", estimated)
            np.savetxt(truth_path, truth)
            original = {"exit_code": 0, "wall_seconds": 12.34, "peak_rss_mib": 56.78,
                        "processing_p95_ms": 9.87, "processed_frames": 5,
                        "evaluation_error": "old failure"}
            (output / "result.json").write_text(json.dumps(original))
            command = [sys.executable, str(Path(benchmark.__file__)), "--evaluate-only",
                       "--output", str(root / "results"), "--data", str(root / "data"),
                       "--mode", "geometry", "--sequences", "shield1"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output / "result.json").read_text())
            for key in ("wall_seconds", "peak_rss_mib", "processing_p95_ms", "processed_frames"):
                self.assertEqual(result[key], original[key])
            self.assertNotIn("evaluation_error", result)
            self.assertLess(result["ate_rmse_m"], 1e-12)
            self.assertEqual(result["matched_samples"], 5)


if __name__ == "__main__":
    unittest.main()
