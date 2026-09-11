#ifndef BIEVR_LIO_INTENSITY_PROCESSING_H_
#define BIEVR_LIO_INTENSITY_PROCESSING_H_

#include "bievr_lio/common.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bievr {

struct IntensityConfig {
  bool enabled = true;
  std::string projection = "spherical";
  int width = 1024;
  int height = 128;
  double vertical_fov_deg = 180.0;
  double scale = 140.0;
  double raw_scale = 1.0;
  int window_width = 41;
  int window_height = 7;
  std::vector<int> pixel_shift_by_row;
  bool line_removal = false;
  std::vector<double> highpass;
  std::vector<double> lowpass;
  double photo_scale = 0.003;
  size_t max_voxels = 100;
  double downsample_resolution = 0.1;
};

struct IntensityFrame {
  Intensities values;
  // A zero return can be valid; missing/invalid measurements are marked here.
  std::vector<uint8_t> valid;
};

// Validate normalization/projection parameters before receiving the first scan.
// Disabled intensity accepts unused settings; sampling controls are validated by their owner.
void validateIntensityConfig(const IntensityConfig& config);

// Operates in the original LiDAR frame and preserves input point order.
// Ouster LUT mode requires the unfiltered organized, staggered input cloud.
// Throws std::invalid_argument for invalid enabled configurations or LUT layout.
IntensityFrame normalizeIntensity(const StampedIntensityPointcloud& cloud,
                                  const IntensityConfig& config,
                                  double min_range, double max_range);

}  // namespace bievr

#endif  // BIEVR_LIO_INTENSITY_PROCESSING_H_
