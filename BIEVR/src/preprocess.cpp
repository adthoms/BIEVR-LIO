#include "bievr_lio/preprocess.h"

#include <Eigen/Eigenvalues>
#include <stdexcept>

#include "unordered_dense/unordered_dense.h"

namespace bievr {

namespace {

struct VoxelDownsampleEntry {
  size_t hash;
  size_t idx;
  double dist;
};

struct VoxelHashIdx {
  size_t hash;
  size_t idx;
};

struct VoxelScore {
  double score;
  size_t hash;
  size_t idx;
};

}  // namespace

void voxelDownsample(const Pointcloud& points_raw, Pointcloud& points_down, double voxel_size,
                     std::vector<size_t>* output_indices) {
  if (!(voxel_size > 0) || !std::isfinite(voxel_size)) {
    throw std::invalid_argument("voxel downsampling resolution must be positive and finite");
  }
  std::vector<VoxelDownsampleEntry> voxel_entries(points_raw.size());

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, points_raw.size()), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t idx = r.begin(); idx != r.end(); ++idx) {
          const Eigen::Vector3i voxel = (points_raw[idx] / voxel_size).array().floor().cast<int>();
          Eigen::Vector3d voxel_center =
              (voxel.cast<double>() + Eigen::Vector3d::Constant(0.5)) * voxel_size;
          double dist = (points_raw[idx] - voxel_center).squaredNorm();
          voxel_entries[idx] = {hashIndexVoxel(voxel.matrix()), idx, dist};
        }
      });

  // Sort by voxel hash, then by distance to voxel center (closest first).
  tbb::parallel_sort(voxel_entries.begin(), voxel_entries.end(),
                     [](const VoxelDownsampleEntry& a, const VoxelDownsampleEntry& b) {
                       return std::tie(a.hash, a.dist, a.idx) < std::tie(b.hash, b.dist, b.idx);
                     });

  std::vector<size_t> selected_indices;
  selected_indices.reserve(points_raw.size());

  size_t curr_hash = std::numeric_limits<size_t>::max();
  for (const auto& entry : voxel_entries) {
    if (entry.hash != curr_hash) {
      selected_indices.push_back(entry.idx);
      curr_hash = entry.hash;
    }
  }

  points_down.resize(selected_indices.size());
  if (output_indices) *output_indices = selected_indices;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, selected_indices.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        points_down[i] = points_raw[selected_indices[i]];
                      }
                    });
}

void sampleInformed(const BIEVRMap& map, const Transform& T_W_L, const Pointcloud& points_raw,
                    Pointcloud& points_coarse, Pointcloud& points_fine, double voxel_size,
                    size_t n_samples, std::vector<size_t>* coarse_indices,
                    std::vector<size_t>* fine_indices) {
  points_fine.clear();
  if (coarse_indices) coarse_indices->clear();
  if (fine_indices) fine_indices->clear();
  if (points_raw.size() <= n_samples) {
    points_coarse.resize(points_raw.size());
    points_coarse.data().topRows(3) = points_raw.data().topRows(3);
    if (coarse_indices) {
      coarse_indices->resize(points_raw.size());
      std::iota(coarse_indices->begin(), coarse_indices->end(), 0);
    }
    return;
  }

  // Lookup hash for each point based on its transformed position from the registration prior.
  std::vector<VoxelHashIdx> voxel_entries(points_raw.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, points_raw.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t idx = r.begin(); idx != r.end(); ++idx) {
                        Point p_w = T_W_L.linear() * points_raw[idx] + T_W_L.translation();
                        voxel_entries[idx] = {map.hashIndex(p_w), idx};
                      }
                    });

  tbb::parallel_sort(voxel_entries.begin(), voxel_entries.end(),
                     [](const VoxelHashIdx& a, const VoxelHashIdx& b) {
                       // Add tie to ensure deterministic order for points in the same voxel, even
                       // if they have the same hash
                       return std::tie(a.hash, a.idx) < std::tie(b.hash, b.idx);
                     });

  // Extract unique hashes of observed voxels
  std::vector<VoxelHashIdx> unique_voxels;
  unique_voxels.reserve(points_raw.size());
  std::vector<bool> voxel_has_extras;
  voxel_has_extras.reserve(points_raw.size());

  for (size_t i = 0; i < voxel_entries.size(); ++i) {
    if (i == 0 || voxel_entries[i].hash != voxel_entries[i - 1].hash) {
      unique_voxels.push_back(voxel_entries[i]);
      voxel_has_extras.push_back(false);
    } else {
      voxel_has_extras.back() = true;
    }
  }

  std::vector<VoxelScore> voxel_scores(unique_voxels.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, unique_voxels.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t idx = r.begin(); idx != r.end(); ++idx) {
                        const auto& entry = unique_voxels[idx];
                        auto voxel = map.getVoxel(entry.hash);
                        double score =
                            (!voxel || !voxel_has_extras[idx]) ? 0.0 : voxel->mean_img_dist_;
                        voxel_scores[idx] = {score, entry.hash, entry.idx};
                      }
                    });

  // Sort by score, descending.
  tbb::parallel_sort(voxel_scores.begin(), voxel_scores.end(),
                     [](const VoxelScore& a, const VoxelScore& b) { return a.score > b.score; });

  ankerl::unordered_dense::set<size_t> informed_voxels;
  size_t n_select = std::min(n_samples, voxel_scores.size());
  informed_voxels.reserve(n_select);

  for (size_t idx = 0; idx < n_select; ++idx) {
    informed_voxels.insert(voxel_scores[idx].hash);
  }

  // Keep all points in informed voxels
  std::vector<size_t> informed_indices;
  informed_indices.reserve(points_raw.size());
  size_t curr_hash = std::numeric_limits<size_t>::max();
  bool curr_informed = false;
  for (const auto& entry : voxel_entries) {
    if (entry.hash != curr_hash) {
      curr_hash = entry.hash;
      curr_informed = informed_voxels.contains(entry.hash);
    }
    if (curr_informed) {
      informed_indices.push_back(entry.idx);
    }
  }

  points_fine.resize(informed_indices.size());
  if (fine_indices) *fine_indices = informed_indices;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, points_fine.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t idx = r.begin(); idx != r.end(); ++idx) {
                        points_fine[idx] = points_raw[informed_indices[idx]];
                      }
                    });

  // Keep one point per coarse voxel for the rest
  size_t n_informed = informed_voxels.size();
  size_t n_coarse = voxel_scores.size() - n_informed;
  points_coarse.resize(n_coarse);
  if (coarse_indices) {
    coarse_indices->reserve(n_coarse);
    for (size_t i = n_informed; i < voxel_scores.size(); ++i) {
      coarse_indices->push_back(voxel_scores[i].idx);
    }
  }
  tbb::parallel_for(tbb::blocked_range<size_t>(n_informed, voxel_scores.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t idx = r.begin(); idx != r.end(); ++idx) {
                        points_coarse[idx - n_informed] = points_raw[voxel_scores[idx].idx];
                      }
                    });
}

IntensitySamples sampleIntensity(const BIEVRMap& map, const Transform& T_W_L,
                                 const Pointcloud& points_raw,
                                 const std::vector<uint8_t>& valid, size_t max_voxels,
                                 double resolution) {
  if (valid.size() != points_raw.size()) {
    throw std::invalid_argument("intensity validity must match source point count");
  }
  IntensitySamples result;
  if (points_raw.empty() || max_voxels == 0) return result;

  std::vector<VoxelHashIdx> entries;
  entries.reserve(points_raw.size());
  for (size_t i = 0; i < points_raw.size(); ++i) {
    if (!points_raw[i].allFinite()) continue;
    const Point p_W = T_W_L * points_raw[i];
    entries.push_back({map.hashIndex(p_W), i});
  }
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return std::tie(a.hash, a.idx) < std::tie(b.hash, b.idx);
  });

  M3 A = M3::Zero();
  std::vector<VoxelScore> scores;
  for (size_t i = 0; i < entries.size();) {
    size_t end = i + 1;
    while (end < entries.size() && entries[end].hash == entries[i].hash) ++end;
    const Voxel* voxel = map.getVoxel(entries[i].hash);
    if (voxel && voxel->observed_) {
      const V3 normal = voxel->T_C_W_.linear().row(2).transpose();
      if (normal.allFinite()) {
        A.noalias() += normal * normal.transpose();
        bool has_intensity = false;
        for (size_t j = i; j < end; ++j) has_intensity |= valid[entries[j].idx] != 0;
        if (has_intensity) scores.push_back({0, entries[i].hash, entries[i].idx});
      }
    }
    i = end;
  }
  if (!(A.trace() > 0) || scores.empty()) return result;
  const Eigen::SelfAdjointEigenSolver<M3> eig(A);
  if (eig.info() != Eigen::Success) return result;
  const auto& values = eig.eigenvalues();
  const double tolerance = 1e-9 * std::max(1.0, A.trace());
  const int dimensions = (10.0 * values[0] > values[1] || values[1] <= tolerance) ? 2 : 1;
  const auto subspace = eig.eigenvectors().leftCols(dimensions);
  for (auto& entry : scores) {
    const Voxel* voxel = map.getVoxel(entry.hash);
    const V3 u = voxel->T_C_W_.linear().row(0).transpose();
    const V3 v = voxel->T_C_W_.linear().row(1).transpose();
    entry.score = voxel->intensity_score_[0] * (subspace.transpose() * u).norm() +
                  voxel->intensity_score_[1] * (subspace.transpose() * v).norm();
    if (!std::isfinite(entry.score)) entry.score = 0;
  }
  std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.hash < b.hash;
  });
  ankerl::unordered_dense::set<size_t> selected;
  for (const auto& entry : scores) {
    if (selected.size() >= max_voxels) break;
    if (std::isfinite(entry.score) && entry.score > 0) selected.insert(entry.hash);
  }
  result.num_voxels = selected.size();
  std::vector<size_t> source_indices;
  Pointcloud candidates;
  for (const auto& entry : entries) {
    if (selected.contains(entry.hash) && valid[entry.idx]) {
      source_indices.push_back(entry.idx);
    }
  }
  candidates.resize(source_indices.size());
  for (size_t i = 0; i < source_indices.size(); ++i) candidates[i] = points_raw[source_indices[i]];
  Pointcloud downsampled;
  std::vector<size_t> downsample_indices;
  voxelDownsample(candidates, downsampled, resolution, &downsample_indices);
  result.indices.reserve(downsample_indices.size());
  for (size_t i : downsample_indices) result.indices.push_back(source_indices[i]);
  return result;
}

}  // namespace bievr
