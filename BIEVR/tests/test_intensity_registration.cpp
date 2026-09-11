#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "bievr_lio/ls_optimizer.h"

namespace {
void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}
}

int main() {
  using namespace bievr;
  Voxel voxel;
  voxel.intensity_smoothed_.resize(2, 2);
  voxel.intensity_smoothed_ << 0, 2, 3, 7;
  voxel.intensity_weights_ = Eigen::MatrixXf::Ones(2, 2);
  double value, dx, dy;
  require(sampleIntensityAndGradient(&voxel, 0.3, 0.4, value, dx, dy), "valid bilinear sample");
  require(std::abs(value - 2.04) < 1e-10 && std::abs(dx - 2.8) < 1e-10 && std::abs(dy - 3.6) < 1e-10, "bilinear value and analytic gradient");
  voxel.intensity_weights_(0, 1) = 0;
  const double h = 1e-6;
  require(sampleIntensityAndGradient(&voxel, 0.3, 0.4, value, dx, dy), "partially masked sample");
  double plus, minus, unused1, unused2;
  sampleIntensityAndGradient(&voxel, 0.3 + h, 0.4, plus, unused1, unused2);
  sampleIntensityAndGradient(&voxel, 0.3 - h, 0.4, minus, unused1, unused2);
  require(std::abs(dx - (plus - minus) / (2 * h)) < 1e-7, "masked gradient must differentiate normalization");
  sampleIntensityAndGradient(&voxel, 0.3, 0.4 + h, plus, unused1, unused2);
  sampleIntensityAndGradient(&voxel, 0.3, 0.4 - h, minus, unused1, unused2);
  require(std::abs(dy - (plus - minus) / (2 * h)) < 1e-7, "masked y gradient must differentiate normalization");
  voxel.intensity_smoothed_(0, 1) = std::numeric_limits<float>::quiet_NaN();
  require(sampleIntensityAndGradient(&voxel, 0.3, 0.4, value, dx, dy), "masked NaN pixels must not poison interpolation");
  voxel.intensity_weights_(0, 0) = 1;
  voxel.intensity_weights_(1, 0) = 0;
  voxel.intensity_weights_(1, 1) = 0;
  require(sampleIntensityAndGradient(&voxel, 0.3, 0.4, value, dx, dy) && value == 0 && dx == 0 && dy == 0, "valid zero intensity and one-corner support must survive");
  voxel.intensity_weights_.setZero();
  require(!sampleIntensityAndGradient(&voxel, 0.3, 0.4, value, dx, dy), "unsupported appearance must be rejected");
  require(!sampleIntensityAndGradient(&voxel, -0.1, 0.4, value, dx, dy), "out-of-bounds appearance must be rejected");

  BIEVRMap map(BIEVRMap::Config{});
  Pointcloud plane;
  for (int x = 0; x < 7; ++x)
    for (int y = 0; y < 7; ++y) plane.push_back(Point(0.08 + x * 0.05, 0.08 + y * 0.05, 0.2));
  map.integratePoints(plane);
  Voxel* patch = const_cast<Voxel*>(map.getVoxel(map.hashIndex(plane[0])));
  require(patch != nullptr, "test plane must make an observed voxel");
  patch->intensity_weights_ = Eigen::MatrixXf::Ones(patch->bump_smoothed_.rows(), patch->bump_smoothed_.cols());
  patch->intensity_smoothed_.resizeLike(patch->bump_smoothed_);
  for (int y = 0; y < patch->intensity_smoothed_.rows(); ++y)
    for (int x = 0; x < patch->intensity_smoothed_.cols(); ++x)
      patch->intensity_smoothed_(y, x) = 10 * x + 3 * y;
  Intensities observations(plane.size());
  std::vector<uint8_t> flags(plane.size(), 1);
  for (size_t i = 0; i < plane.size(); ++i) {
    const Point projected = patch->T_C_W_ * plane[i];
    observations[i] = (10 * projected.x() + 3 * projected.y()) * map.inv_px_size;
  }
  RegistrationConfig config;
  config.img_jacobian = false;
  config.max_iterations = 15;
  const Point shift = patch->T_C_W_.linear().transpose() * Point(0.015, 0.0045, 0);
  Transform initial(shift);
  LsqRegistration registration(map, plane, config, &observations, &flags);
  const Transform result = registration.computeTransformation(initial);
  require(result.matrix().allFinite(), "joint solve must remain finite");
  require(result.translation().norm() < initial.translation().norm() * 0.2, "appearance must recover a tangential plane displacement");
  require(registration.numPhotometricPoints() > 0, "joint solve must count photometric support");
  require(registration.numEffectivePoints() <= static_cast<int>(plane.size()), "effective support must count unique points");
  std::fill(flags.begin(), flags.end(), 0);
  LsqRegistration fallback(map, plane, config, &observations, &flags);
  LsqRegistration geometry(map, plane, config);
  const Transform fallback_pose = fallback.computeTransformation(initial);
  require(fallback_pose.matrix().isApprox(geometry.computeTransformation(initial).matrix(), 1e-10), "no-appearance solve must match geometry-only fallback");

  // An uninformative geometry point on an image boundary must not pin a correction
  // supported by the remaining textured plane. Leaving that one correspondence
  // cannot improve its cost, but must allow the other 49 residuals to improve.
  patch->bump_weights_.setOnes();
  patch->bump_smoothed_.setZero();
  Pointcloud boundary_plane = plane;
  boundary_plane.push_back(patch->T_C_W_.inverse() * Point(0, 0.25, 0));
  Intensities boundary_observations(boundary_plane.size());
  std::vector<uint8_t> boundary_flags(boundary_plane.size(), 1);
  boundary_flags.back() = 0;
  for (size_t i = 0; i < boundary_plane.size(); ++i) {
    const Point projected = patch->T_C_W_ * boundary_plane[i];
    boundary_observations[i] =
        (10 * (projected.x() - 0.015) + 3 * projected.y()) * map.inv_px_size;
  }
  LsqRegistration boundary_registration(map, boundary_plane, config,
                                       &boundary_observations, &boundary_flags);
  const Transform boundary_pose = boundary_registration.computeTransformation(Transform::Identity());
  const Point boundary_translation = patch->T_C_W_.linear() * boundary_pose.translation();
  require(boundary_translation.x() < -0.012,
          "one uninformative boundary point must not pin the textured-plane correction");

  // The requested intensity only exists beyond the image. A trial cannot improve
  // the objective by leaving the image and silently dropping its photo terms.
  observations.setConstant(10000);
  std::fill(flags.begin(), flags.end(), 1);
  config.img_residual = false;
  config.max_iterations = 8;
  LsqRegistration bounded(map, plane, config, &observations, &flags);
  const Transform bounded_pose = bounded.computeTransformation(Transform::Identity());
  require(bounded_pose.matrix().allFinite(), "out-of-support trial must not produce a nonfinite pose");
  size_t retained_support = 0;
  for (size_t i = 0; i < plane.size(); ++i) {
    const Point projected = patch->T_C_W_ * (bounded_pose * plane[i]);
    retained_support += sampleIntensityAndGradient(patch, projected.x() * map.inv_px_size,
                                                    projected.y() * map.inv_px_size, value, dx, dy);
  }
  require(retained_support >= std::ceil(0.9 * plane.size()),
          "LM must preserve photometric overlap across repeated outer iterations");

  // One dark outlier at u=0 asks for a negative u correction outside the map.
  // The other 49 points are already aligned. Omitting the outlier with zero cost
  // would falsely reward that motion even though every retained point gets worse.
  for (int y = 0; y < patch->intensity_smoothed_.rows(); ++y)
    for (int x = 0; x < patch->intensity_smoothed_.cols(); ++x)
      patch->intensity_smoothed_(y, x) = 100 + 10 * x;
  for (size_t i = 0; i < boundary_plane.size(); ++i) {
    const Point projected = patch->T_C_W_ * boundary_plane[i];
    boundary_observations[i] = 100 + 10 * projected.x() * map.inv_px_size;
  }
  boundary_observations[boundary_plane.size() - 1] = 0;
  boundary_flags.back() = 1;
  LsqRegistration outlier_registration(map, boundary_plane, config,
                                      &boundary_observations, &boundary_flags);
  const Transform outlier_pose = outlier_registration.computeTransformation(Transform::Identity());
  require(outlier_pose.matrix().isApprox(Transform::Identity().matrix(), 1e-6),
          "dropping an outlier cannot reward a motion that worsens every retained residual");

  Pointcloud empty;
  LsqRegistration empty_registration(map, empty, config);
  require(empty_registration.computeTransformation(initial).matrix().isApprox(initial.matrix()),
          "empty geometry fallback must preserve the prior");
  return 0;
}
