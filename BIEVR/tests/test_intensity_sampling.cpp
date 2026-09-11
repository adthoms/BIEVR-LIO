#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>

#include "bievr_lio/preprocess.h"

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
  Pointcloud cloud;
  cloud.resize(4);
  cloud[0] = Point(0.01, 0, 0);
  cloud[1] = Point(1.01, 0, 0);
  cloud[2] = Point(1.04, 0, 0);
  cloud[3] = Point(2.0, 0, 0);
  Pointcloud filtered, sampled;
  std::vector<size_t> indices;
  filterMinMaxRange(cloud, filtered, 0.5, 1.5, &indices);
  require(indices == std::vector<size_t>({1, 2}), "range filtering must retain source indices");
  voxelDownsample(filtered, sampled, 0.1, &indices);
  require(indices == std::vector<size_t>({1}), "downsampling must retain the chosen source index");
  require(sampled[0].isApprox(filtered[indices[0]]), "downsampling point/index alignment");

  BIEVRMap map(BIEVRMap::Config{});
  Pointcloud plane;
  for (int x = 0; x < 4; ++x)
    for (int y = 0; y < 4; ++y) plane.push_back(Point(0.1 + x * 0.06, 0.1 + y * 0.06, 0.2));
  map.integratePoints(plane);
  Voxel* voxel = const_cast<Voxel*>(map.getVoxel(map.hashIndex(plane[0])));
  require(voxel != nullptr, "test plane must make an observed voxel");
  voxel->intensity_score_ = Eigen::Vector2d(10, 4);
  std::vector<uint8_t> valid(plane.size(), 1);
  auto selection = sampleIntensity(map, Transform::Identity(), plane, valid, 100, 0.1);
  require(selection.num_voxels == 1 && !selection.indices.empty(), "texture in a plane must constrain its null space");
  require(std::set<size_t>(selection.indices.begin(), selection.indices.end()).size() == selection.indices.size(), "sampling must not duplicate source points");
  const auto original = selection.indices;
  voxel->T_C_W_.linear().row(0) *= -1;
  voxel->T_C_W_.linear().row(2) *= -1;
  require(sampleIntensity(map, Transform::Identity(), plane, valid).indices == original, "eigenvector and normal signs must not affect selection");
  std::fill(valid.begin(), valid.end(), 0);
  require(sampleIntensity(map, Transform::Identity(), plane, valid).indices.empty(), "invalid intensity must not be sampled");
  std::fill(valid.begin(), valid.end(), 1);
  voxel->intensity_score_.setZero();
  require(sampleIntensity(map, Transform::Identity(), plane, valid).indices.empty(), "flat appearance must not add samples");

  Pointcloud coarse, fine;
  fine.resize(1);
  std::vector<size_t> coarse_indices, fine_indices{99};
  sampleInformed(map, Transform::Identity(), plane, coarse, fine, 0.1, 100, &coarse_indices, &fine_indices);
  require(fine.empty() && fine_indices.empty(), "small-cloud sampling must clear reused fine outputs");
  require(coarse_indices.size() == plane.size(), "small-cloud sampling must report all indices");
  return 0;
}
