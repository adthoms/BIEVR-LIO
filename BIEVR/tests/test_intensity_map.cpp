#include "bievr_lio/bievr_map.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, const std::string& message, double tolerance = 1e-4) {
  require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          message + ": expected " + std::to_string(expected) + ", got " +
              std::to_string(actual));
}

bievr::Pointcloud corners() {
  bievr::Pointcloud cloud;
  cloud.resize(4);
  cloud[0] = bievr::Point(0.1, 0.1, 0.2);
  cloud[1] = bievr::Point(0.3, 0.1, 0.2);
  cloud[2] = bievr::Point(0.1, 0.3, 0.2);
  cloud[3] = bievr::Point(0.3, 0.3, 0.2);
  return cloud;
}

template <class Map>
auto voxelAt(const Map& map, const bievr::Point& point) {
  const auto* voxel = map.getVoxel(map.hashIndex(point));
  require(voxel != nullptr, "voxel must have enough geometry to be observed");
  return voxel;
}

template <class Map>
void accumulation() {
  typename Map::Config config;
  config.weighted = true;
  Map map(config);
  const auto cloud = corners();
  bievr::Intensities intensities(1, 4);
  intensities << 0.0, 40.0, 80.0, 120.0;
  std::vector<double> ranges{1.0, 4.0, 8.0, 10.0};
  require(map.integratePoints(cloud, &ranges, &intensities), "initial appearance integration");
  const auto* voxel = voxelAt(map, cloud[0]);
  require(voxel->intensity_img_.size() > 0, "valid appearance allocates image");
  for (int i = 0; i < cloud.size(); ++i) {
    const bievr::Point p = voxel->T_C_W_ * cloud[i];
    const int x = std::lround(p.x() / config.px_size);
    const int y = std::lround(p.y() / config.px_size);
    near(voxel->intensity_img_(y, x), intensities[i], "first intensity, including valid zero");
    near(voxel->intensity_weights_(y, x), std::min(0.5, 1.0 / ranges[i]), "capped range weight");
  }
  intensities.array() += 100.0;
  for (auto& range : ranges) range *= 2.0;
  require(map.integratePoints(cloud, &ranges, &intensities), "second appearance integration");
  voxel = voxelAt(map, cloud[0]);
  for (int i = 0; i < cloud.size(); ++i) {
    const bievr::Point p = voxel->T_C_W_ * cloud[i];
    const int x = std::lround(p.x() / config.px_size);
    const int y = std::lround(p.y() / config.px_size);
    const double previous_weight = std::min(0.5, 2.0 / ranges[i]);
    const double new_weight = std::min(0.5, 1.0 / ranges[i]);
    const double expected = ((intensities[i] - 100.0) * previous_weight +
                             intensities[i] * new_weight) / (previous_weight + new_weight);
    near(voxel->intensity_img_(y, x), expected,
         "repeated observations use a weighted mean");
  }
}

template <class Map>
void pendingAndMissing() {
  Map map(typename Map::Config{});
  const auto cloud = corners();
  bievr::Pointcloud first(cloud.data().leftCols(2));
  bievr::Pointcloud second(cloud.data().rightCols(2));
  bievr::Intensities initial(1, 2);
  initial << 0.0, 40.0;
  require(map.integratePoints(first, nullptr, &initial), "pending appearance integration");
  require(map.getVoxel(map.hashIndex(cloud[0])) == nullptr, "two points remain unobserved");
  require(map.integratePoints(second), "geometry completes the voxel");
  const auto* voxel = voxelAt(map, cloud[0]);
  near(voxel->intensity_weights_.sum(), 2.0, "pending appearance survives geometry-only scan");
  for (int i = 0; i < first.size(); ++i) {
    const bievr::Point p = voxel->T_C_W_ * first[i];
    near(voxel->intensity_img_(std::lround(p.y() / map.pixel_size),
                               std::lround(p.x() / map.pixel_size)), initial[i],
         "pending intensity stays associated with its point");
  }
  const auto old_img = voxel->intensity_img_;
  const auto old_weights = voxel->intensity_weights_;
  bievr::Intensities invalid(1, 4);
  invalid << 100.0, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), 100.0;
  std::vector<uint8_t> validity{0, 1, 1, 0};
  require(map.integratePoints(cloud, nullptr, &invalid, &validity), "missing appearance keeps geometry");
  voxel = voxelAt(map, cloud[0]);
  require(voxel->intensity_img_.isApprox(old_img) && voxel->intensity_weights_.isApprox(old_weights),
          "invalid appearance cannot dilute accumulated intensity");
}

template <class Map>
void geometryOnlyAndValidation() {
  const auto cloud = corners();
  Map baseline(typename Map::Config{});
  Map missing(typename Map::Config{});
  require(baseline.integratePoints(cloud), "baseline integration");
  bievr::Intensities intensities = bievr::Intensities::Constant(1, 4, 80.0);
  std::vector<uint8_t> invalid(4, 0);
  require(missing.integratePoints(cloud, nullptr, &intensities, &invalid), "masked input integration");
  const auto* baseline_voxel = voxelAt(baseline, cloud[0]);
  const auto* missing_voxel = voxelAt(missing, cloud[0]);
  require(baseline_voxel->intensity_img_.size() == 0 &&
              missing_voxel->intensity_img_.size() == 0,
          "geometry-only voxels do not allocate appearance images");
  require(baseline_voxel->bump_img_.isApprox(missing_voxel->bump_img_, 0.0) &&
              baseline_voxel->bump_weights_.isApprox(missing_voxel->bump_weights_, 0.0) &&
              baseline_voxel->T_C_W_.matrix().isApprox(missing_voxel->T_C_W_.matrix(), 0.0),
          "missing appearance preserves geometric map exactly");
  Map bad(typename Map::Config{});
  bievr::Intensities short_intensities(1, 3);
  require(!bad.integratePoints(cloud, nullptr, &short_intensities), "reject intensity size mismatch");
  std::vector<uint8_t> short_validity(3, 1);
  require(!bad.integratePoints(cloud, nullptr, &intensities, &short_validity),
          "reject validity size mismatch");
  require(bad.size() == 0, "mismatched metadata cannot partially mutate map");
}

template <class Map>
void reprojection() {
  typename Map::Config config;
  config.norm_tol_deg = 1.0;
  Map map(config);
  bievr::Pointcloud cloud;
  cloud.resize(81);
  bievr::Intensities intensities(1, 81);
  std::vector<uint8_t> validity(81);
  for (int y = 0; y < 9; ++y) {
    for (int x = 0; x < 9; ++x) {
      const int i = y * 9 + x;
      cloud[i] = bievr::Point(0.05 + x * 0.05, 0.05 + y * 0.05, 0.25);
      intensities[i] = 1.0 + i;
      validity[i] = i % 2 != 0;
    }
  }
  require(map.integratePoints(cloud, nullptr, &intensities, &validity), "appearance before rotation");
  const auto before = *voxelAt(map, cloud[0]);
  bievr::Pointcloud tilted;
  tilted.resize(810);
  for (int i = 0; i < tilted.size(); ++i) {
    const bievr::Point p = cloud[i % cloud.size()];
    tilted[i] = bievr::Point(p.x(), p.y(), 0.10 + 0.6 * p.x());
  }
  require(map.integratePoints(tilted), "geometry rotates the image plane");
  const auto* after = voxelAt(map, cloud[0]);
  require(!before.T_C_W_.linear().isApprox(after->T_C_W_.linear(), 1e-3),
          "fixture must change the voxel normal");
  Eigen::MatrixXf expected_img = Eigen::MatrixXf::Zero(after->bump_img_.rows(), after->bump_img_.cols());
  Eigen::MatrixXf expected_weights = expected_img;
  Eigen::MatrixXi projected = Eigen::MatrixXi::Zero(expected_img.rows(), expected_img.cols());
  const bievr::Transform transform = after->T_C_W_ * before.T_C_W_.inverse();
  int collisions = 0;
  int missing_winners = 0;
  for (int y = 0; y < before.bump_img_.rows(); ++y) {
    for (int x = 0; x < before.bump_img_.cols(); ++x) {
      if (before.bump_weights_(y, x) <= 0.0) continue;
      const bievr::Point p = transform * bievr::Point(x * config.px_size, y * config.px_size,
                                                     before.bump_img_(y, x));
      const int px = std::lround(p.x() / config.px_size);
      const int py = std::lround(p.y() / config.px_size);
      if (px < 0 || py < 0 || px >= expected_img.cols() || py >= expected_img.rows()) continue;
      if (projected(py, px)) {
        ++collisions;
        if (before.intensity_weights_(y, x) == 0.0 && expected_weights(py, px) > 0.0) {
          ++missing_winners;
        }
      }
      projected(py, px) = 1;
      expected_img(py, px) = before.intensity_img_(y, x);
      expected_weights(py, px) = before.intensity_weights_(y, x);
    }
  }
  require(collisions > 0 && missing_winners > 0,
          "fixture exercises missing-intensity collision winners (collisions=" +
              std::to_string(collisions) + ", missing winners=" + std::to_string(missing_winners) + ")");
  require(after->intensity_img_.isApprox(expected_img, 1e-6) &&
              after->intensity_weights_.isApprox(expected_weights, 1e-6),
          "reprojected appearance follows the same surface and collision winner as height");
}

template <class Map>
void smoothingAndScore() {
  typename Map::Config config;
  config.smooth = true;
  Map map(config);
  bievr::Pointcloud cloud;
  cloud.resize(81);
  bievr::Intensities intensities(1, 81);
  for (int y = 0; y < 9; ++y) {
    for (int x = 0; x < 9; ++x) {
      const int i = y * 9 + x;
      cloud[i] = bievr::Point(0.05 + x * 0.05, 0.05 + y * 0.05, 0.25);
      intensities[i] = 40.0;
    }
  }
  require(map.integratePoints(cloud, nullptr, &intensities), "constant intensity patch");
  const auto* voxel = voxelAt(map, cloud[0]);
  near(voxel->intensity_score_.norm(), 0.0, "constant texture has no information");
  for (int y = 0; y < voxel->intensity_img_.rows(); ++y) {
    for (int x = 0; x < voxel->intensity_img_.cols(); ++x) {
      if (voxel->intensity_weights_(y, x) > 0.0) {
        near(voxel->intensity_smoothed_(y, x), 40.0, "masked smoothing preserves constants");
      }
    }
  }
  for (int i = 0; i < cloud.size(); ++i) {
    const bievr::Point p = voxel->T_C_W_ * cloud[i];
    intensities[i] = 40.0 + 100.0 * p.x();
  }
  require(map.integratePoints(cloud, nullptr, &intensities), "directional texture update");
  voxel = voxelAt(map, cloud[0]);
  require(voxel->intensity_score_.x() > 50.0, "varying image direction carries information");
  near(voxel->intensity_score_.y(), 0.0, "constant image direction has no information", 2e-3);
  require(!voxel->intensity_img_.isApprox(voxel->intensity_smoothed_, 1e-5),
          "enabled smoothing processes changed appearance and neighbors");

  const bievr::Point center = voxel->T_C_W_ * cloud[40];
  const int x = std::lround(center.x() / config.px_size);
  const int y = std::lround(center.y() / config.px_size);
  const double neighbor_raw = voxel->intensity_img_(y, x + 1);
  const double neighbor_smoothed = voxel->intensity_smoothed_(y, x + 1);
  bievr::Pointcloud one_point;
  one_point.resize(1);
  one_point[0] = cloud[40];
  bievr::Intensities bright = bievr::Intensities::Constant(1, 1, 400.0);
  require(map.integratePoints(one_point, nullptr, &bright), "single pixel intensity update");
  voxel = voxelAt(map, cloud[0]);
  near(voxel->intensity_img_(y, x + 1), neighbor_raw, "neighbor has no new raw measurement");
  require(voxel->intensity_smoothed_(y, x + 1) > neighbor_smoothed + 1.0,
          "a changed intensity pixel also refreshes its smoothed neighbors");
}

template <class Map>
void run() {
  Map single_voxel(typename Map::Config{});
  require(single_voxel.integratePoints(corners()) && single_voxel.size() == 1,
          "a cloud entirely inside one voxel must integrate that voxel");
  accumulation<Map>();
  pendingAndMissing<Map>();
  geometryOnlyAndValidation<Map>();
  reprojection<Map>();
  smoothingAndScore<Map>();
}

}  // namespace

int main() {
  try {
    run<bievr::BIEVRMap>();
    std::cout << "intensity map tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
