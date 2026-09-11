#include "bievr_lio/intensity_processing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace bievr {
namespace {

int wrap(int64_t value, int width) {
  value %= width;
  return static_cast<int>(value < 0 ? value + width : value);
}

// OpenCV's default BORDER_REFLECT_101, used by the original COIN-LIO filter.
int reflect(int value, int height) {
  if (height == 1) return 0;
  const int period = 2 * (height - 1);
  value = wrap(value, period);
  return value < height ? value : period - value;
}

bool validKernel(const std::vector<double>& kernel) {
  return !kernel.empty() && kernel.size() % 2 == 1 &&
         std::all_of(kernel.begin(), kernel.end(),
                     [](double value) { return std::isfinite(value); });
}

void validate(const IntensityConfig& config) {
  if (config.projection != "spherical" && config.projection != "ouster_lut") {
    throw std::invalid_argument("intensity.projection must be spherical or ouster_lut");
  }
  if (config.width <= 0 || config.height <= 0 ||
      config.width > 32768 || config.height > 32768 ||
      static_cast<int64_t>(config.width) * config.height > 16777216 ||
      !std::isfinite(config.vertical_fov_deg) || config.vertical_fov_deg <= 0 ||
      config.vertical_fov_deg > 180 || !std::isfinite(config.scale) || config.scale <= 0 ||
      !std::isfinite(config.raw_scale) || config.raw_scale <= 0 ||
      config.window_width <= 0 || config.window_height <= 0 ||
      config.window_width % 2 != 1 || config.window_height % 2 != 1) {
    throw std::invalid_argument("invalid intensity image or normalization parameters");
  }
  if (config.projection == "ouster_lut" &&
      config.pixel_shift_by_row.size() != static_cast<size_t>(config.height)) {
    throw std::invalid_argument("intensity.pixel_shift_by_row must contain one shift per row");
  }
  if (config.line_removal &&
      (config.projection != "ouster_lut" || !validKernel(config.highpass) ||
       !validKernel(config.lowpass))) {
    throw std::invalid_argument("intensity line removal requires Ouster LUT and finite odd kernels");
  }
}

// A missing return must not manufacture a line artifact. Apply the correction
// only where the separable filter's complete support has been observed.
std::vector<double> lineCorrection(const std::vector<double>& image,
                                   const std::vector<size_t>& counts,
                                   const IntensityConfig& config) {
  const int width = config.width;
  const int height = config.height;
  std::vector<double> vertical(image.size(), 0.0);
  std::vector<uint8_t> valid(image.size(), 0);
  const int ry = static_cast<int>(config.highpass.size() / 2);
  const int rx = static_cast<int>(config.lowpass.size() / 2);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t index = static_cast<size_t>(y) * width + x;
      bool observed = true;
      for (int k = -ry; k <= ry; ++k) {
        const size_t neighbor = static_cast<size_t>(reflect(y + k, height)) * width + x;
        observed = observed && counts[neighbor] != 0;
        vertical[index] += config.highpass[k + ry] * image[neighbor];
      }
      valid[index] = observed && std::isfinite(vertical[index]);
    }
  }
  std::vector<double> correction(image.size(), 0.0);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t index = static_cast<size_t>(y) * width + x;
      bool observed = true;
      double filtered = 0.0;
      for (int k = -rx; k <= rx; ++k) {
        const size_t neighbor = static_cast<size_t>(y) * width + wrap(x + k, width);
        observed = observed && valid[neighbor];
        filtered += config.lowpass[k + rx] * vertical[neighbor];
      }
      if (observed && std::isfinite(filtered)) correction[index] = filtered;
    }
  }
  return correction;
}

// Integral images keep sparse normalization O(image pixels + input points).
class Brightness {
 public:
  Brightness(const std::vector<double>& image, const std::vector<size_t>& counts,
             int width, int height)
      : width_(width), stride_(width + 1),
        sum_(static_cast<size_t>(width + 1) * (height + 1), 0.0),
        count_(sum_.size(), 0.0) {
    for (int y = 0; y < height; ++y) {
      double row_sum = 0.0;
      double row_count = 0.0;
      for (int x = 0; x < width; ++x) {
        const size_t source = static_cast<size_t>(y) * width + x;
        row_sum += image[source];
        row_count += counts[source] > 0 ? 1 : 0;
        const size_t target = static_cast<size_t>(y + 1) * stride_ + x + 1;
        sum_[target] = sum_[target - stride_] + row_sum;
        count_[target] = count_[target - stride_] + row_count;
      }
    }
  }

  double mean(int x, int y, const IntensityConfig& config) const {
    const int y0 = std::max<int64_t>(0, static_cast<int64_t>(y) - config.window_height / 2);
    const int y1 = std::min<int64_t>(config.height,
                                   static_cast<int64_t>(y) + config.window_height / 2 + 1);
    if (config.window_width >= width_) return ratio(0, width_, y0, y1);
    const int start = wrap(static_cast<int64_t>(x) - config.window_width / 2, width_);
    const int end = start + config.window_width;
    double sum = rect(sum_, start, std::min(end, width_), y0, y1);
    double count = rect(count_, start, std::min(end, width_), y0, y1);
    if (end > width_) {
      sum += rect(sum_, 0, end - width_, y0, y1);
      count += rect(count_, 0, end - width_, y0, y1);
    }
    return count > 0 ? std::max(0.0, sum / count) : 0.0;
  }

 private:
  double rect(const std::vector<double>& data, int x0, int x1, int y0, int y1) const {
    return data[static_cast<size_t>(y1) * stride_ + x1] -
           data[static_cast<size_t>(y1) * stride_ + x0] -
           data[static_cast<size_t>(y0) * stride_ + x1] +
           data[static_cast<size_t>(y0) * stride_ + x0];
  }

  double ratio(int x0, int x1, int y0, int y1) const {
    const double count = rect(count_, x0, x1, y0, y1);
    return count > 0 ? std::max(0.0, rect(sum_, x0, x1, y0, y1) / count) : 0.0;
  }

  int width_;
  int stride_;
  std::vector<double> sum_;
  std::vector<double> count_;
};

}  // namespace

void validateIntensityConfig(const IntensityConfig& config) {
  if (config.enabled) validate(config);
}

IntensityFrame normalizeIntensity(const StampedIntensityPointcloud& cloud,
                                  const IntensityConfig& config,
                                  double min_range, double max_range) {
  IntensityFrame frame{Intensities::Zero(cloud.size()),
                       std::vector<uint8_t>(cloud.size(), 0)};
  if (!config.enabled) return frame;
  validateIntensityConfig(config);
  if (!std::isfinite(min_range) || min_range < 0 ||
      !std::isfinite(max_range) || max_range <= min_range) {
    throw std::invalid_argument("invalid intensity range limits");
  }
  if (cloud.empty() || !cloud.has_intensity) return frame;

  const int width = config.width;
  const int height = config.height;
  const size_t pixels = static_cast<size_t>(width) * height;
  const bool ouster = config.projection == "ouster_lut";
  if (ouster && (cloud.size() != pixels ||
                 cloud.scan_width != static_cast<size_t>(width) ||
                 cloud.scan_height != static_cast<size_t>(height))) {
    throw std::invalid_argument("Ouster intensity LUT requires a complete organized input cloud");
  }
  std::vector<double> image(pixels, 0.0);
  std::vector<size_t> counts(pixels, 0);
  std::vector<size_t> point_pixels(cloud.size(), 0);
  const double fov = config.vertical_fov_deg * std::numbers::pi / 180.0;

  for (size_t i = 0; i < cloud.size(); ++i) {
    const Point point = cloud.data().col(i).head<3>();
    const double raw = cloud.intensities()(i);
    const double intensity = raw * config.raw_scale;
    const double range = point.stableNorm();
    if (!point.allFinite() || !std::isfinite(raw) || raw < 0 ||
        !std::isfinite(intensity) || !std::isfinite(range) || range <= 0 ||
        range < min_range || range > max_range) continue;

    int x;
    int y;
    if (ouster) {
      y = static_cast<int>(i / width);
      x = wrap(static_cast<int64_t>(i % width) + config.pixel_shift_by_row[y], width);
    } else {
      const double elevation = std::asin(std::clamp(point.z() / range, -1.0, 1.0));
      if (std::abs(elevation) > fov / 2.0 + 1e-12) continue;
      x = wrap(static_cast<int64_t>(std::floor(
                   -width * std::atan2(point.y(), point.x()) / (2.0 * std::numbers::pi) +
                   width / 2.0)), width);
      y = std::clamp(static_cast<int>(std::floor(-height * elevation / fov + height / 2.0)),
                     0, height - 1);
    }
    const size_t pixel = static_cast<size_t>(y) * width + x;
    point_pixels[i] = pixel;
    frame.values(i) = intensity;
    frame.valid[i] = 1;
    // Stable online mean also prevents overflow when several returns collide.
    ++counts[pixel];
    image[pixel] += (intensity - image[pixel]) / static_cast<double>(counts[pixel]);
  }

  if (config.line_removal) {
    const auto correction = lineCorrection(image, counts, config);
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
      if (counts[pixel]) image[pixel] = std::max(0.0, image[pixel] - correction[pixel]);
    }
    for (size_t i = 0; i < cloud.size(); ++i) {
      if (frame.valid[i]) frame.values(i) = std::max(0.0, frame.values(i) - correction[point_pixels[i]]);
    }
  }

  const Brightness brightness(image, counts, width, height);
  for (size_t i = 0; i < cloud.size(); ++i) {
    if (!frame.valid[i]) continue;
    const int x = static_cast<int>(point_pixels[i] % width);
    const int y = static_cast<int>(point_pixels[i] / width);
    const double value = config.scale * (frame.values(i) / (brightness.mean(x, y, config) + 1.0));
    if (!std::isfinite(value)) {
      frame.valid[i] = 0;
      frame.values(i) = 0.0;
    } else {
      frame.values(i) = std::clamp(value, 0.0, 255.0);
    }
  }
  return frame;
}

}  // namespace bievr
