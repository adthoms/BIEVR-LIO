# COIN-BIEVR

COIN-BIEVR adds LiDAR intensity constraints to BIEVR-LIO's oriented voxel
images, following the [COIN-BIEVR paper](https://icra2026-rigorous-perception.github.io/pdf/pfreundschuh2026.pdf).
Intensity processing is enabled by default. ROS1 and ROS2 use the same CPU
implementation in the ROS-independent core, with Eigen and TBB; no GPU or
OpenCV dependency is added.

This independent implementation has not reproduced the paper's reported ATE.
The [comparison and audit](#historical-measurements-and-paper-comparison) record
the measured gaps and the Ouster filter correction.

## Configuration and input

Existing launch commands continue to work. For example:

```bash
ros2 launch bievr_lio_ros2 process_bag.launch.py \
  sensor_config:=enwide rosbag:=/path/to/rosbag2_directory rviz:=false
```

To run geometry alone, copy `config/params.yaml`, change the following leaf,
and pass the copy with `params:=/absolute/path/geometry.yaml`:

```yaml
intensity:
  enabled: false
```

The main intensity settings in `config/params.yaml` are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `enabled` | `true` | Enable intensity normalization, sampling, and registration. |
| `scale` | `140.0` | Scale intensity after local brightness normalization. |
| `window_width`, `window_height` | `41`, `7` | Brightness normalization window in pixels. |
| `photo_scale` | `0.003` | Scale the intensity residual before robust optimization. |
| `max_voxels` | `100` | Maximum number of textured voxels selected per scan. |
| `downsample_resolution_m` | `0.1` | Spatial downsampling resolution for intensity samples. |

ENWIDE and Newer College sensor configurations load
`config/intensity/ouster_enwide.yaml` and `ouster_ncd.yaml`, respectively.
These profiles specify 1024 × 128 Ouster projection, per-row pixel shifts,
raw intensity scaling of 0.25, and line removal. Profiles that enable line
removal also load the neighboring `line_removal.yaml` filter coefficients.
The separable line filter operates on zero-filled missing pixels; filtering
does not make a missing return valid. Horizontal boundaries wrap and vertical
boundaries reflect. Requiring a fully occupied filter neighborhood caused
abrupt correction changes and has been removed.
Profile paths resolve relative to the YAML file containing `intensity.profile`.
The loader merges profile defaults first, then explicit input YAML leaves, so
an explicit setting such as `intensity.enabled: false` takes precedence.

Ouster projection requires the original organized, staggered cloud with its
row/column layout intact. Apply range filtering and motion compensation after
intensity normalization. Other configurations, including `geode_gamma`, use
spherical projection by default: 1024 × 128 pixels, 180° vertical field of view,
raw scaling 1.0, and no line removal. These are configurable sensor assumptions.
The GEODE calibration and evaluation convention is described below.

Missing or nonfinite intensity has separate validity metadata; a measured zero
remains valid. Points without appearance support can still contribute geometry.
An invalid enabled projection configuration fails configuration loading.
An incompatible Ouster scan layout reports a warning and uses geometry for that
scan. Disabled intensity skips profile-file loading. Raw intensity remains
available for cloud publication.

The Ouster pixel shifts and line-removal coefficients come from the COIN-LIO
reference configuration; attribution is retained under `config/intensity/`.
These parameters and the selected defaults have **not** established an exact
reproduction of the paper's reported results.

The paper does not specify the photometric residual weight. The initial
COIN-LIO reference value, `0.00095`, failed late in the local TunnelS sequence.
The default is therefore `0.003`, used globally across all four historical
dataset runs below. These sequences were also used to select the weight; validation on
additional recordings remains useful for new sensor configurations.

## Estimation and numerical details

1. Project raw returns into an intensity image, average collisions, optionally
   remove Ouster line artifacts, and normalize by masked local brightness.
   Normalized values are clipped to [0, 255] and retain their source indices.
2. Store appearance beside each voxel's height image. Appearance has its own
   validity and accumulated weights. Normal changes reproject both channels;
   appearance follows the corresponding height collision winner.
3. Form the normal matrix from unique observed voxels. Rank texture along the
   least-constrained translation directions, select up to 100 positive-scoring
   voxels, and downsample their valid source points at 0.1 m. Merge these indices
   with the geometric samples without duplicating points.
4. Optimize height and photometric residuals together. The intensity residual is
   `photo_scale * (observed_intensity - map_intensity)`, with the existing Huber
   loss and a Jacobian for the optimizer's right-multiplied pose update.

Several numerical choices make the implementation explicit:

- Eigenvalues are ordered increasingly. Select the two weakest directions when
  `10 * lambda1 > lambda2` or `lambda2 <= 1e-9 * max(1, trace(A))`; otherwise select
  the weakest direction. Use projection norms into that subspace to weight each
  image-axis texture score. This avoids dependence on eigenvector signs or an
  arbitrary basis within a degenerate eigenspace. An empty normal matrix adds
  no appearance samples.
- Photometric interpolation differentiates the masked, normalized bilinear
  interpolant exactly, including its denominator near missing pixels.
- During each LM damping loop, retain the voxel and geometry/appearance
  membership chosen by the outer linearization. If a residual loses image
  support, retain its cost from that linearization instead of removing its
  contribution. Accept only a positive cost reduction with at least 90% of
  geometric and photometric support retained independently, measured against
  both the current and first usable joint linearizations. Refresh
  correspondences at the next outer linearization. This permits limited
  movement across patch boundaries without treating discarded residuals as
  an improvement. A solve with no usable photometric constraints uses the
  geometric path.
- Effective-point diagnostics count unique source points. Photometric support
  is counted separately. Geometric image Jacobians are explicitly zero when
  disabled, and empty solves preserve the prior.

The map also fixes an existing single-voxel grouping defect: a cloud whose
points all belong to one voxel now initializes that voxel correctly. Therefore,
the feature build with intensity disabled includes this geometry correction;
retain a separate baseline build when measuring the complete change.

## Build and test

From the repository root, build the core and run its CTest suite:

```bash
cmake -S BIEVR -B /tmp/coin-bievr-core \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build /tmp/coin-bievr-core --parallel 4
ctest --test-dir /tmp/coin-bievr-core --output-on-failure
```

The tests cover intensity normalization and projection, configuration profiles,
appearance accumulation and reprojection, source-index sampling, masked
interpolation derivatives, planar registration, missing support, and geometry
fallback. Core tests also require yaml-cpp for configuration checks.

Run the evaluator's numerical and recorded-result regression tests from the
repository root:

```bash
python3 scripts/test_benchmark_coin_bievr.py
```

For ROS2, run from the workspace root. Replace `kilted` with the installed ROS2
distribution as needed; the existing README includes Jazzy setup instructions.

```bash
source /opt/ros/kilted/setup.bash
colcon build --packages-up-to bievr_lio_ros2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
ctest --test-dir build/bievr_lio --output-on-failure
ctest --test-dir build/bievr_lio_ros2 --output-on-failure
```

## Dataset comparison

`scripts/benchmark_coin_bievr.py` replays bags offline and writes a separate
directory for each mode and sequence. It requires Python NumPy, PyYAML,
`/usr/bin/time`, and the ROS2 MCAP storage plugin.

Keep the baseline isolated from the feature build. For the baseline used when
this work started, run the following from the feature repository:

```bash
coin_repo="$PWD"
baseline_ws=/tmp/bievr-baseline-ws
mkdir -p "$baseline_ws/src"
git worktree add --detach "$baseline_ws/src/BIEVR-LIO" \
  bc66e07ef1aa5ff22ce41206fc5ce63356e8ca64
cd "$baseline_ws"
source /opt/ros/kilted/setup.bash
colcon build --packages-up-to bievr_lio_ros2 \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
cd "$coin_repo"
```

With the feature workspace built as above, compare TunnelD first. Set the
feature install path to the workspace's `install` directory:

```bash
feature_install=/absolute/path/to/ros2_ws/install
benchmark_output=/tmp/coin-bievr-results
python3 scripts/benchmark_coin_bievr.py \
  --install /tmp/bievr-baseline-ws/install --output "$benchmark_output" \
  --mode baseline --threads 8 --sequences tunnel_d
python3 scripts/benchmark_coin_bievr.py \
  --install "$feature_install" --output "$benchmark_output" \
  --mode geometry --threads 8 --sequences tunnel_d
python3 scripts/benchmark_coin_bievr.py \
  --install "$feature_install" --output "$benchmark_output" \
  --mode coin --threads 8 --sequences tunnel_d
```

`baseline` and `geometry` both disable intensity; the install prefix selects
which implementation runs. `coin` enables intensity. The runner uses the
feature repository's current geometry parameters for all modes and saves the
actual algorithm YAML. Sensor configurations are read from
`config/sensor_configs/`; archive those files separately when reproducing runs.
Run each build in a clean shell to avoid importing another build's
library paths. Use a fresh output directory for repeated measurements because
the same mode/sequence path is overwritten.

Use `--photo-scale` for a controlled residual-weight comparison. `--solver-debug`
enables `optimization.lm_debug_print`: per-trial costs, retained support,
damping, and separate geometric/photometric information along the weakest
geometric direction. Frame headers include the scan timestamp and pose prior.
These diagnostics are disabled by default.

Recompute evaluation from a finished run without replaying its bag:

```bash
python3 scripts/benchmark_coin_bievr.py --evaluate-only \
  --output "$benchmark_output" --mode baseline --sequences tunnel_d
```

This updates `result.json` and preserves recorded runtime and memory fields;
it does not require an install prefix or change the saved trajectory or logs.

`--data` defaults to `~/data/data`; `--ros-setup` defaults to
`/opt/ros/kilted/setup.bash`. Omitting `--sequences` runs all four supported
sequences: `tunnel_d`, `tunnel_s`, `shield1`, and `quad_hard`.

| Dataset | Local input under the data root | Availability |
| --- | --- | --- |
| ENWIDE TunnelD | `enwide/tunnel_d/2023-08-08-17-50-31-tunnel_d` | MCAP, 119 s, prism GT. |
| ENWIDE TunnelS | `enwide/tunnel_s/2023-08-08-17-12-37-tunnel_s` | MCAP, 238 s, prism GT. |
| GEODE Shield1 | `geode/Shield_tunnel1_gamma_pc2` | MCAP, 543 s, irregular Livox points and position GT. |
| NCD QuadHard | `newer_college/2021-ouster-os0-128-alphasense/c1_newer_college/2021-07-01-11-35-14_0-quad-hard_ros2` | MCAP, 188 s, TUM GT. |
| NCD Cloister | Two `2021-12-02-10-15-59_0-cloister_ros2` and `2021-12-02-10-19-05_1-cloister_ros2` segments under `c2_newer_college` | Available locally; not included in the runner. |
| GEODE FlatSurfacesS | — | Unavailable locally. |

The Shield1 PointCloud2 conversion avoids the original ROS1 Livox message type
inside the converted ROS2 bag. Its `/livox/lidar` and `/livox/imu` topics match
`geode_gamma`. ENWIDE uses `/ouster/points` and `/ouster/imu`; Newer College uses
`/os_cloud_node/points` and `/os_cloud_node/imu`.

Each run saves `params.yaml`, `trajectory.tum`, `replay.log`, `resources.txt`,
and `result.json`. Results include exit status, translation ATE RMSE and maximum,
matched samples and coverage, processing latency summaries, throughput, wall
time, peak RSS, and photometric activity when available in the logs.

Evaluation version 2 interpolates estimated comparison positions at actual GT
timestamps. It uses only brackets at most 0.2 s apart, accepts exact timestamp
matches, and never extrapolates or interpolates missing GT. The paired positions
are aligned with a rigid translation/rotation fit without scale.

Convert estimated IMU poses to the dataset's comparison origin before
interpolation and alignment:

- ENWIDE uses prism positions with
  `T_IMU_PRISM.translation = [-0.006253, 0.011775, 0.10825]`, from the released
  `prism_imu_extrinsics.txt`.
- Newer College multi-camera GT is expressed in the
  [Base frame](https://ori-drs.github.io/newer-college-dataset/ground-truth/).
  Compose the released `os_imu_lidar_transforms.yaml` transforms and use
  `T_IMU_BASE.translation = [0.013, -0.012, -0.106]`.
- GEODE Shield1 uses the official
  [gamma-to-GT transform](https://github.com/PengYu-Team/GEODE_dataset/blob/main/script/gamma2GT_leica.py):
  `T_eval = T_device * inverse(T)`, where `T` has translation
  `[0.00947221, -0.308202, -0.365733]` and quaternion **wxyz**
  `[0.999901, -0.00492765, 0.00575961, 0.0117651]`. GT orientation columns are
  unused, so the provided zero quaternions do not affect position evaluation.

The GEODE correction treats the logged `T_W_I` as the configured device frame.
The existing `geode_gamma` sensor configuration combines the official
external-device extrinsic with `/livox/imu`; this convention remains identical
across compared modes and is recorded in each result's evaluation caveat.

`ground_truth_match_fraction` is the fraction of GT timestamps inside the
estimated time span that have valid interpolation support.
`ground_truth_time_coverage` reports overlapping time divided by estimated
duration. These distinguish sparse GT sampling from missing trajectory time.
For example, Shield1 GT has a median sampling interval of 0.36 s and ends about
22 s before the local bag ends. Inspect frame completion, GT time coverage, and
matched fractions alongside ATE; a small error on a partial trajectory does not
establish successful completion.

## Historical measurements and paper comparison

The original baseline is commit `bc66e07ef1aa5ff22ce41206fc5ce63356e8ca64`; the feature
implementation is commit `67dd0bb84bef2bc9677a9e5f02faae877fe809f0` on
`feature/coin-bievr`, with `photo_scale: 0.003`. Both used Release
builds with GCC 13.3 and ROS2 Kilted on an Intel Core Ultra 9 285H, with 16 logical
CPUs available and an 8-thread estimator limit. These original four-sequence
measurements use evaluation version 2 and predate the Ouster filter correction.
Both the original and current GEODE configuration pair `/livox/imu` with the
external-device extrinsic and normalize over 180°. Its recorded trajectory is
evaluated in that configured external-device frame, with the caveat above.

| Sequence | Baseline translation ATE RMSE (m) | COIN-BIEVR translation ATE RMSE (m) |
| --- | ---: | ---: |
| ENWIDE TunnelD | 50.928569 | 0.254177 |
| ENWIDE TunnelS | 213.930295 | 0.988403 |
| GEODE Shield1 | 0.485710 | 0.486326 |
| NCD QuadHard | 0.057096 | 0.057115 |

In these historical runs, the ENWIDE errors decreased substantially. The measured changes on Shield1
and QuadHard were below 0.001 m. With intensity disabled, the feature's QuadHard
trajectory matched the baseline exactly, including its 0.057095618 m ATE.
Replaying TunnelS before the Ouster filter correction, using the original
default configuration, also produced a byte-identical trajectory to that run.

The original isolated build and normal ROS workspace build both passed all six
core CTest suites and the ROS2 conversion suite. The current ROS workspace
build also passes those seven suites and all six Python evaluator tests.

| Sequence | Mean processing (ms) | p95 processing (ms) | Replay throughput (Hz) | Peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| ENWIDE TunnelD | 43.409 | 47.424 | 19.571 | 179.5 |
| ENWIDE TunnelS | 43.476 | 48.269 | 19.055 | 202.7 |
| GEODE Shield1 | 11.516 | 14.057 | 80.164 | 356.1 |
| NCD QuadHard | 48.292 | 56.529 | 17.149 | 492.9 |

Processing times cover estimator steps; replay throughput includes bag reading
and process overhead. These are historical, uncontended single-run measurements.
The concurrent diagnostic runs from the later audit are not fair timing
comparisons. Every original run exited
successfully and processed more than 99.8% of recorded LiDAR frames. Photometric
constraints were active in more than 99.9% of reported registration frames.
Shield1 GT covers 95.95% of the estimated time span; its final approximately 22 s
remain unscored. The GEODE calibration caveat described above applies.

The paper reports the following ATE values in
[Table I](https://icra2026-rigorous-perception.github.io/pdf/pfreundschuh2026.pdf):

| Sequence and method | Original local ATE RMSE (m) | Current branch (m) | Paper ATE RMSE (m) |
| --- | ---: | ---: | ---: |
| TunnelD, COIN-BIEVR | 0.254177 | 0.297320 | 0.369 |
| TunnelS, COIN-BIEVR | 0.988403 | 0.500880 | 0.432 |
| Shield1, COIN-BIEVR | 0.486326 | 0.486326 | 0.220 |
| Shield1, geometric BIEVR baseline | 0.485710 | 0.485710 | 0.256 |
| QuadHard, COIN-BIEVR | 0.057115 | 0.054362 | 0.049 |

The current implementation is commit
`4d3419bdd8fba21f120e8a6825c046bd16efdb82`, including the Ouster filter correction.
The global photometric weight remains 0.003. The GEODE calibration and evaluator
retain their original behavior.

The original local runs do not reproduce the paper's TunnelS or Shield1 accuracy.
The Shield1 geometric baseline also differs, so its gap cannot be attributed
solely to intensity registration. The sensor mismatch above is one concrete
comparability defect. The paper does not specify its timestamp association
protocol; using the official GEODE nearest-neighbor association with a 0.1 s
threshold changes the old Shield1 ATE by less than 0.0006 m, which does not explain
the reported gap.

The adopted Ouster filter correction removes the extra complete-neighborhood
gate and applies the COIN-LIO line-filter response over zero-filled missing
pixels. Missing returns remain invalid. With the same global weight, the full
TunnelS run improved from 0.988403 m to 0.500880 m ATE. TunnelD changed from
0.254177 m to 0.297320 m, and QuadHard improved from 0.057115 m to 0.054362 m.
All three runs completed with more than 99.8% frame coverage. GEODE does not
enable this Ouster filter, so its original result remains 0.486326 m.
The saved Ouster audit artifacts carry version-3 metadata from the experimental
runner; re-evaluating their trajectories with the retained version-2 evaluator
reproduces all three ATE values exactly.

This establishes a consequential implementation difference, but the remaining
TunnelS and Shield1 accuracy gaps are unresolved. The paper does not publish
the residual weight or full preprocessing and evaluation configuration needed
to claim an exact numerical reproduction. The sampler's sign-invariant scoring
and the LM support policy described above also remain explicit implementation
choices.

Recorded artifacts are under the workspace root:

- `.cache/coin-bievr/results/baseline/<sequence>/`: baseline runs.
- `.cache/coin-bievr/weight-003/coin/<sequence>/`: original runs with the adopted
  global weight, including exact YAML, trajectories, logs, and `result.json`.
- `.cache/coin-bievr/results/geometry/quad_hard/`: intensity-disabled regression.
- `.cache/coin-bievr/final-default/coin/tunnel_s/`: final default-configuration replay.
- `.cache/coin-bievr/paper-gap/line-filter-results/coin/`: isolated Ouster
  filter-support comparisons.

The historical validation covered TunnelD, TunnelS, Shield1, and QuadHard. Cloister
was not replayed, and FlatSurfacesS was unavailable locally. These measurements
establish behavior on the available recordings, without claiming exact
reproduction of the paper's experimental setup or results.
