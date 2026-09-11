// This Levenberg-Marquardt Optimizer builds upon the code used in the nanogicp module in DLIO
// https://github.com/vectr-ucla/direct_lidar_inertial_odometry

#include "bievr_lio/ls_optimizer.h"

#include <Eigen/Eigenvalues>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>
#include <stdexcept>

#include "bievr_lio/log++.h"
#include "bievr_lio/utils.h"

namespace bievr {

LsqRegistration::LsqRegistration(const BIEVRMap& map, const Pointcloud& source,
                                 const RegistrationConfig& config,
                                 const Intensities* intensities,
                                 const std::vector<uint8_t>* photometric_flags)
    : config_(config), map_(map), points_j_(source), intensities_(intensities),
      photometric_flags_(photometric_flags) {
  if ((intensities && static_cast<size_t>(intensities->size()) != source.size()) ||
      (photometric_flags && photometric_flags->size() != source.size())) {
    throw std::invalid_argument("photometric channels must match registration source points");
  }
  if (intensities && config.photo_scale > 0 && std::isfinite(config.photo_scale)) {
    for (size_t i = 0; i < source.size(); ++i) {
      if ((!photometric_flags || (*photometric_flags)[i]) && std::isfinite((*intensities)[i])) {
        coin_mode_ = true;
        break;
      }
    }
  }
}

Transform LsqRegistration::computeTransformation(const Transform& T_W_L_init) {
  Transform x0 = T_W_L_init;
  converged_ = false;
  lm_lambda_ = -1.0;
  num_effective_points_ = 0;
  num_photometric_points_ = 0;
  initial_geometry_support_ = initial_photo_support_ = 0;
  current_geometry_support_ = current_photo_support_ = 0;
  frozen_joint_support_ = false;
  debug_weak_axis_.setZero();

  skew_points_j_.resize(points_j_.size());
  tbb::parallel_for(tbb::blocked_range<size_t>(0, points_j_.size()),
                    [&](const tbb::blocked_range<size_t>& r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        skew_points_j_[i] = skew(points_j_[i]);
                      }
                    });

  if (config_.lm_debug_print) {
    LOG(I, "***************** optimize *****************");
  }

  for (int i = 0; i < config_.max_iterations && !converged_; i++) {
    Transform delta;
    if (!stepLm(x0, delta)) {
      LOG(W, "lm not converged!!");
      break;
    }
    converged_ = isConverged(delta);
  }

  return x0;
}

bool LsqRegistration::isConverged(const Transform& delta) const {
  Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / config_.rotation_epsilon * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / config_.transformation_epsilon * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

double LsqRegistration::linearize(const Transform& T_W_L, Matrix66* H, Vector6* b) {
  if (H && b) {
    if (coin_mode_) {
      const double value = linearizeJoint(T_W_L, H, b);
      frozen_joint_support_ = num_photometric_points_ > 0;
      if (frozen_joint_support_) return value;
    }
    return linearizeGeometry(T_W_L, H, b);
  }
  if (config_.lm_debug_print) debug_overlap_rejected_ = false;
  return frozen_joint_support_ ? linearizeJoint(T_W_L, nullptr, nullptr)
                               : linearizeGeometry(T_W_L, nullptr, nullptr);
}

double LsqRegistration::linearizeJoint(const Transform& T_W_L, Matrix66* H, Vector6* b) {
  const bool compute_jacobians = H && b;
  if (compute_jacobians) correspondences_.assign(points_j_.size(), {});
  Accumulator identity;
  identity.huber_delta = config_.huber_delta;
  const Accumulator total = tbb::parallel_deterministic_reduce(
      tbb::blocked_range<size_t>(0, points_j_.size()), identity,
      [&](const tbb::blocked_range<size_t>& range, Accumulator local) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          Correspondence& support = correspondences_[i];
          const Point p_W = T_W_L * points_j_[i];
          if (!p_W.allFinite()) {
            local.support_lost |= !compute_jacobians && (support.geometry || support.photo);
            continue;
          }
          if (compute_jacobians) {
            size_t hash = map_.hashIndex(p_W);
            support.voxel = map_.getVoxel(hash);
            if (!support.voxel && map_.nearestVoxel(p_W, hash)) support.voxel = map_.getVoxel(hash);
          }
          if (!support.voxel || (!compute_jacobians && !support.geometry && !support.photo)) continue;
          const auto& T_C_W = support.voxel->T_C_W_;
          const Point p_C = T_C_W * p_W;
          const double x = p_C.x() * map_.inv_px_size, y = p_C.y() * map_.inv_px_size;
          double height = 0, hx = 0, hy = 0;
          bool geometry = false;
          if (compute_jacobians || support.geometry) {
            geometry = !config_.img_residual ||
                       (compute_jacobians ? sampleValueAndGradient(support.voxel, x, y, height, hx, hy)
                                          : getSubPixelValue(support.voxel, x, y, height));
          }
          double intensity = 0, ix = 0, iy = 0;
          const bool photo_requested = compute_jacobians
              ? ((!photometric_flags_ || (*photometric_flags_)[i]) && std::isfinite((*intensities_)[i]))
              : support.photo;
          const bool photo = photo_requested &&
              sampleIntensityAndGradient(support.voxel, x, y, intensity, ix, iy);
          if (!compute_jacobians) {
            // A missing residual retains exactly its baseline loss. Dropping observations
            // therefore cannot create a cost reduction, while one boundary point cannot
            // prevent a correction supported by the rest of the scan.
            if (support.geometry && !geometry) local.error_sum += support.geometry_cost;
            if (support.photo && !photo) local.error_sum += support.photo_cost;
          }
          if (compute_jacobians) {
            support.geometry = geometry;
            support.photo = photo;
          }
          Eigen::Matrix<double, 3, 6> pose_jacobian;
          if (compute_jacobians && (geometry || photo)) {
            const Rotation R_C_L = T_C_W.linear() * T_W_L.linear();
            pose_jacobian.leftCols<3>().noalias() = -R_C_L * skew_points_j_[i];
            pose_jacobian.rightCols<3>() = R_C_L;
          }
          if (geometry) {
            Row6 J;
            if (compute_jacobians) {
              Eigen::RowVector2d image_jacobian = Eigen::RowVector2d::Zero();
              if (config_.img_jacobian) image_jacobian << hx * map_.inv_px_size, hy * map_.inv_px_size;
              J = pose_jacobian.row(2) - image_jacobian * pose_jacobian.topRows<2>();
            }
            const double residual = p_C.z() - height;
            local.add(residual, compute_jacobians ? &J : nullptr);
            if (compute_jacobians) support.geometry_cost = local.loss(residual);
            ++local.geometry_count;
          }
          if (photo) {
            Row6 J;
            if (compute_jacobians) {
              J = -config_.photo_scale * map_.inv_px_size *
                  Eigen::RowVector2d(ix, iy) * pose_jacobian.topRows<2>();
            }
            const double residual = config_.photo_scale * ((*intensities_)[i] - intensity);
            local.add(residual, compute_jacobians ? &J : nullptr);
            if (compute_jacobians) support.photo_cost = local.loss(residual);
            ++local.photo_count;
          }
          if (geometry || photo) ++local.unique_count;
        }
        return local;
      }, [](Accumulator a, const Accumulator& b) { a.merge(b); return a; });
  if (compute_jacobians) {
    *H = total.H;
    *b = total.b;
    num_effective_points_ = total.unique_count;
    num_photometric_points_ = total.photo_count;
    current_geometry_support_ = total.geometry_count;
    current_photo_support_ = total.photo_count;
    if (total.photo_count > 0 && initial_photo_support_ == 0) {
      initial_geometry_support_ = total.geometry_count;
      initial_photo_support_ = total.photo_count;
    }
    if (config_.lm_debug_print && total.photo_count > 0) logJointInformation(T_W_L);
  } else {
    // Require 90% overlap in each modality, both for this linearization and for the
    // first usable joint linearization. The latter prevents gradual support erosion
    // across outer iterations. Cost substitution handles isolated boundary losses;
    // this guard rejects steps leaving most of the modeled surface behind.
    constexpr double kMinimumSupportFraction = 0.9;
    if (total.geometry_count < kMinimumSupportFraction *
                                   std::max(initial_geometry_support_, current_geometry_support_) ||
        total.photo_count < kMinimumSupportFraction *
                                std::max(initial_photo_support_, current_photo_support_)) {
      if (config_.lm_debug_print) {
        debug_overlap_rejected_ = true;
        LOG(I, "LM overlap retained_geo=" << total.geometry_count
            << " reference_geo=" << std::max(initial_geometry_support_, current_geometry_support_)
            << " retained_photo=" << total.photo_count
            << " reference_photo=" << std::max(initial_photo_support_, current_photo_support_));
      }
      return std::numeric_limits<double>::infinity();
    }
  }
  return total.support_lost ? std::numeric_limits<double>::infinity() : total.error_sum;
}

void LsqRegistration::logJointInformation(const Transform& pose) {
  // This diagnostic pass runs only when requested and does not feed the normal equations.
  M3 normals = M3::Zero(), geometry_H = M3::Zero(), photo_H = M3::Zero();
  V3 geometry_b = V3::Zero(), photo_b = V3::Zero();
  double geometry_cost = 0.0, photo_cost = 0.0;
  for (size_t i = 0; i < correspondences_.size(); ++i) {
    const auto& support = correspondences_[i];
    if (!support.voxel || (!support.geometry && !support.photo)) continue;
    const Point point = support.voxel->T_C_W_ * (pose * points_j_[i]);
    const Rotation rotation = support.voxel->T_C_W_.linear() * pose.linear();
    const double x = point.x() * map_.inv_px_size, y = point.y() * map_.inv_px_size;
    const auto accumulate = [&](double residual, const Eigen::RowVector3d& J,
                                M3& information, V3& gradient) {
      const double magnitude = std::abs(residual);
      const double weight = magnitude <= config_.huber_delta ? 1.0 : config_.huber_delta / magnitude;
      information.noalias() += weight * J.transpose() * J;
      gradient.noalias() += weight * J.transpose() * residual;
    };
    if (support.geometry) {
      double height = 0.0, hx = 0.0, hy = 0.0;
      if (!config_.img_residual ||
          sampleValueAndGradient(support.voxel, x, y, height, hx, hy)) {
        Eigen::RowVector3d J = rotation.row(2);
        if (config_.img_jacobian) {
          J -= map_.inv_px_size * Eigen::RowVector2d(hx, hy) * rotation.topRows<2>();
        }
        normals.noalias() += rotation.row(2).transpose() * rotation.row(2);
        accumulate(point.z() - height, J, geometry_H, geometry_b);
        geometry_cost += support.geometry_cost;
      }
    }
    if (support.photo) {
      double intensity = 0.0, ix = 0.0, iy = 0.0;
      if (sampleIntensityAndGradient(support.voxel, x, y, intensity, ix, iy)) {
        const Eigen::RowVector3d J = -config_.photo_scale * map_.inv_px_size *
                                    Eigen::RowVector2d(ix, iy) * rotation.topRows<2>();
        accumulate(config_.photo_scale * ((*intensities_)[i] - intensity), J, photo_H, photo_b);
        photo_cost += support.photo_cost;
      }
    }
  }
  const Eigen::SelfAdjointEigenSolver<M3> eigen(normals);
  if (eigen.info() != Eigen::Success) return;
  debug_weak_axis_ = eigen.eigenvectors().col(0);
  V3 world_axis = pose.linear() * debug_weak_axis_;
  Eigen::Index dominant;
  world_axis.cwiseAbs().maxCoeff(&dominant);
  if (world_axis[dominant] < 0) {
    debug_weak_axis_ = -debug_weak_axis_;
    world_axis = -world_axis;
  }
  LOG(I, "LM joint geo=" << current_geometry_support_ << " photo=" << current_photo_support_
      << " costs_geo_photo=" << geometry_cost << "," << photo_cost
      << " weak_world=" << world_axis.transpose()
      << " normal_eigenvalues=" << eigen.eigenvalues().transpose()
      << " weak_H_geo_photo=" << debug_weak_axis_.dot(geometry_H * debug_weak_axis_)
      << "," << debug_weak_axis_.dot(photo_H * debug_weak_axis_)
      << " weak_b_geo_photo=" << debug_weak_axis_.dot(geometry_b)
      << "," << debug_weak_axis_.dot(photo_b));
}

double LsqRegistration::linearizeGeometry(const Transform& T_W_L, Matrix66* H, Vector6* b) {
  const bool compute_jacobians = (H != nullptr && b != nullptr);

  Accumulator identity_accumulator;
  identity_accumulator.huber_delta = config_.huber_delta;

  Accumulator total = tbb::parallel_deterministic_reduce(
      tbb::blocked_range<size_t>(0, points_j_.size()),
      identity_accumulator,  // identity
      [&](const tbb::blocked_range<size_t>& r, Accumulator local_acc) -> Accumulator {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          Point p_W = T_W_L.linear() * points_j_[i] + T_W_L.translation();
          size_t hash = map_.hashIndex(p_W);
          const Voxel* voxel = map_.getVoxel(hash);
          if (!voxel) {
            if (!map_.nearestVoxel(p_W, hash)) continue;
            voxel = map_.getVoxel(hash);
            if (!voxel) continue;
          }

          const double inv_size = map_.inv_px_size;
          const auto& T_C_W = voxel->T_C_W_;

          Rotation R_o_j = T_C_W.linear() * T_W_L.linear();
          const Point p_o = T_C_W * p_W;

          const double x = p_o.x() * inv_size;
          const double y = p_o.y() * inv_size;

          double I = 0.0;

          if (!compute_jacobians) {
            if (config_.img_residual) {
              if (!getSubPixelValue(voxel, x, y, I)) continue;
            }
            const double r = p_o.z() - I;
            local_acc.add(r, nullptr);
            continue;
          }

          double dIdx = 0.0;
          double dIdy = 0.0;
          if (config_.img_residual) {
            if (!sampleValueAndGradient(voxel, x, y, I, dIdx, dIdy)) continue;
          }

          Eigen::Matrix<double, 3, 6> SE3_Jac;
          SE3_Jac.block<3, 3>(0, 3) = R_o_j;  // d p_o / d t
          SE3_Jac.block<3, 3>(0, 0).noalias() = -R_o_j * skew_points_j_[i];

          Eigen::Matrix<double, 1, 2> I_Jac = Eigen::Matrix<double, 1, 2>::Zero();
          if (config_.img_jacobian) {
            I_Jac(0, 0) = dIdx;
            I_Jac(0, 1) = dIdy;
            I_Jac *= inv_size;
          }

          Row6 J = SE3_Jac.row(2) - (I_Jac * SE3_Jac.topRows<2>());

          const double r = p_o.z() - I;
          local_acc.add(r, &J);
        }

        return local_acc;
      },
      [](const Accumulator& a, const Accumulator& b) -> Accumulator {
        Accumulator out = a;
        out.merge(b);
        return out;
      });

  if (compute_jacobians) {
    *H = total.H;
    *b = total.b;
    // Remember how many points contributed correspondences in this (Jacobian)
    // linearization so the pipeline can report the effective point count.
    num_effective_points_ = total.count;
    num_photometric_points_ = 0;
  }

  return total.error_sum;
}

namespace {

Eigen::Quaterniond so3_exp(const Eigen::Vector3d& omega) {
  double theta_sq = omega.dot(omega);

  double theta;
  double imag_factor;
  double real_factor;
  if (theta_sq < 1e-10) {
    theta = 0;
    double theta_quad = theta_sq * theta_sq;
    imag_factor = 0.5 - 1.0 / 48.0 * theta_sq + 1.0 / 3840.0 * theta_quad;
    real_factor = 1.0 - 1.0 / 8.0 * theta_sq + 1.0 / 384.0 * theta_quad;
  } else {
    theta = std::sqrt(theta_sq);
    double half_theta = 0.5 * theta;
    imag_factor = std::sin(half_theta) / theta;
    real_factor = std::cos(half_theta);
  }

  return Eigen::Quaterniond(real_factor, imag_factor * omega.x(), imag_factor * omega.y(),
                            imag_factor * omega.z());
}

}  // namespace

bool LsqRegistration::stepLm(Transform& x0, Transform& delta) {
  Matrix66 H;
  Vector6 b;
  double y0 = linearize(x0, &H, &b);
  if (!H.allFinite() || !b.allFinite() || !std::isfinite(y0)) return false;
  if (b.squaredNorm() == 0) {
    if (config_.lm_debug_print) LOG(I, "LM stationary cost=" << y0 << " gradient=0");
    delta.setIdentity();
    return true;
  }

  if (lm_lambda_ < 0.0) {
    lm_lambda_ = std::max(1e-12, config_.lm_init_lambda_factor * H.diagonal().array().abs().maxCoeff());
  }

  double nu = 2.0;
  for (int i = 0; i < config_.lm_max_iterations; i++) {
    Eigen::LDLT<Matrix66> solver(H + lm_lambda_ * Matrix66::Identity());
    Vector6 d = solver.solve(-b);
    if (!d.allFinite()) return false;
    delta.setIdentity();
    delta.linear() = so3_exp(d.head<3>()).toRotationMatrix();
    delta.translation() = d.tail<3>();

    Transform xi = x0 * delta;
    double yi = linearize(xi);
    double rho = (y0 - yi) / (d.dot(lm_lambda_ * d - b));

    const bool rejected = !std::isfinite(yi) || !std::isfinite(rho) ||
                          (frozen_joint_support_ ? rho <= 0 : rho < 0);
    if (config_.lm_debug_print) {
      const char* decision = !rejected ? "accept" : debug_overlap_rejected_ ? "reject_overlap"
          : (!std::isfinite(yi) || !std::isfinite(rho)) ? "reject_nonfinite" : "reject_cost";
      LOG(I, "LM trial=" << i << " joint=" << frozen_joint_support_ << " decision=" << decision
          << " costs_base_trial=" << y0 << "," << yi << " rho=" << rho
          << " damping=" << lm_lambda_ << " translation=" << d.tail<3>().norm()
          << " rotation=" << d.head<3>().norm()
          << " weak_translation=" << debug_weak_axis_.dot(d.tail<3>()));
    }
    if (rejected) {
      if (std::isfinite(yi) && std::isfinite(rho) && isConverged(delta)) {
        return true;
      }

      lm_lambda_ *= nu;
      nu *= 2;
      continue;
    }

    x0 = xi;
    lm_lambda_ *= std::max(1.0 / 3.0, 1 - std::pow(2 * rho - 1, 3));
    return true;
  }

  return false;
}

}  // namespace bievr
