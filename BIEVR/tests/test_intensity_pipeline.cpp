#include "bievr_lio/pipeline.h"

#include <iostream>

int main() {
  bievr::Pipeline::Config config;
  config.intensity.projection = "ouster_lut";
  config.intensity.width = 4;
  config.intensity.height = 2;
  config.intensity.pixel_shift_by_row = {0, 0};
  config.intensity.window_width = 3;
  config.intensity.window_height = 1;
  bievr::Pipeline pipeline(config);
  bievr::StampedIntensityPointcloud cloud;
  cloud.resize(8);
  cloud.scan_width = 8;
  cloud.scan_height = 1;  // An incomplete/flattened scan does not match the factory layout.
  cloud.stamp = 1000000000;
  cloud.end_stamp = 1100000000;
  for (size_t i = 0; i < cloud.size(); ++i) cloud[i] << 2.0, 0.1 * i, 0.0, 0.01 * i, 100.0;
  bievr::ImuMeasurement imu{cloud.stamp, bievr::V3(0, 0, 9.81), bievr::V3::Zero()};
  try {
    pipeline.processFrame({imu}, cloud);
  } catch (const std::invalid_argument&) {
    std::cerr << "unusable image layout must preserve geometry processing\n";
    return 1;
  }
  std::cout << "intensity pipeline fallback passed\n";
}
