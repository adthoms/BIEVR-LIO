#include "bievr_lio/intensity_processing.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using bievr::IntensityConfig;
using bievr::StampedIntensityPointcloud;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, const char* message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-8) {
    std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
    throw std::runtime_error(message);
  }
}

IntensityConfig smallConfig() {
  IntensityConfig config;
  config.width = 8;
  config.height = 4;
  config.window_width = 3;
  config.window_height = 3;
  return config;
}

StampedIntensityPointcloud cloudOf(size_t count) {
  StampedIntensityPointcloud cloud;
  cloud.resize(count);
  cloud.data().setZero();
  cloud.data().row(0).setConstant(2.0);
  return cloud;
}

void sparseBrightnessIgnoresEmptyPixels() {
  auto cloud = cloudOf(1);
  cloud.intensities()(0, 0) = 100.0;
  const auto frame = bievr::normalizeIntensity(cloud, smallConfig(), 0.1, 100.0);
  require(frame.valid == std::vector<uint8_t>{1}, "single valid return lost");
  near(frame.values(0), 140.0 * 100.0 / 101.0, "empty pixels diluted brightness");
}

void collidingPixelsPreserveIndividualReturns() {
  auto cloud = cloudOf(2);
  cloud.intensities() << 10.0, 30.0;
  const auto frame = bievr::normalizeIntensity(cloud, smallConfig(), 0.1, 100.0);
  near(frame.values(0), 140.0 * 10.0 / 21.0, "first colliding return changed");
  near(frame.values(1), 140.0 * 30.0 / 21.0, "second colliding return changed");
  near(cloud.intensities()(0, 1), 30.0, "normalization modified raw intensity");
}

void zeroIntensityRemainsAnObservedPixel() {
  auto cloud = cloudOf(2);
  cloud.intensities() << 0.0, 20.0;
  const auto frame = bievr::normalizeIntensity(cloud, smallConfig(), 0.1, 100.0);
  require(frame.valid == std::vector<uint8_t>({1, 1}), "zero return marked absent");
  near(frame.values(0), 0.0, "zero return changed");
  near(frame.values(1), 140.0 * 20.0 / 11.0, "zero return excluded from brightness");
}

void invalidReturnsDoNotContaminateNeighborhoods() {
  auto cloud = cloudOf(6);
  cloud.intensities() << 100.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
      10000.0, 10000.0, 10000.0;
  cloud.data()(0, 3) = 200.0;
  cloud.data()(0, 4) = 0.0;
  cloud.data()(0, 5) = std::numeric_limits<double>::infinity();
  const auto frame = bievr::normalizeIntensity(cloud, smallConfig(), 0.1, 100.0);
  require(frame.valid == std::vector<uint8_t>({1, 0, 0, 0, 0, 0}),
          "invalid return accepted");
  near(frame.values(0), 140.0 * 100.0 / 101.0, "invalid return polluted brightness");
  for (int i = 1; i < frame.values.cols(); ++i) near(frame.values(i), 0.0, "invalid output nonzero");
}

void sphericalNeighborhoodWrapsAcrossAzimuthSeam() {
  auto cloud = cloudOf(2);
  cloud.data().col(0).head<3>() << -2.0, 0.01, 0.0;
  cloud.data().col(1).head<3>() << -2.0, -0.01, 0.0;
  cloud.intensities() << 10.0, 30.0;
  const auto frame = bievr::normalizeIntensity(cloud, smallConfig(), 0.1, 100.0);
  near(frame.values(0), 140.0 * 10.0 / 21.0, "left seam did not wrap");
  near(frame.values(1), 140.0 * 30.0 / 21.0, "right seam did not wrap");
}

void rawScalingAndSaturation() {
  auto cloud = cloudOf(2);
  cloud.intensities() << 0.0, 100.0;
  auto config = smallConfig();
  config.raw_scale = 0.25;
  const auto frame = bievr::normalizeIntensity(cloud, config, 0.1, 100.0);
  near(frame.values(1), 255.0, "normalized intensity not saturated");
  cloud.resize(1);
  cloud.intensities()(0, 0) = 4.0;
  near(bievr::normalizeIntensity(cloud, config, 0.1, 100.0).values(0),
       70.0, "raw intensity scale ignored");
}

void emptyAndDisabledClouds() {
  auto config = smallConfig();
  const auto empty = bievr::normalizeIntensity(cloudOf(0), config, 0.1, 100.0);
  require(empty.values.size() == 0 && empty.valid.empty(), "empty cloud changed size");
  config.enabled = false;
  const auto disabled = bievr::normalizeIntensity(cloudOf(2), config, 0.1, 100.0);
  require(disabled.valid == std::vector<uint8_t>({0, 0}), "disabled feature produced intensity");
}

void invalidConfigurationIsRejected() {
  auto config = smallConfig();
  config.window_width = 0;
  bool rejected = false;
  try { bievr::normalizeIntensity(cloudOf(1), config, 0.1, 100.0); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "zero normalization window accepted");
}

void absentIntensityDisablesOnlyTheIntensityChannel() {
  auto cloud = cloudOf(1);
  cloud.intensities()(0, 0) = 42.0;
  cloud.has_intensity = false;
  const auto frame = bievr::normalizeIntensity(cloud, smallConfig(), 0.1, 100.0);
  require(frame.valid == std::vector<uint8_t>{0}, "missing intensity field was accepted");
  near(frame.values(0), 0.0, "missing intensity field produced a measurement");
}

IntensityConfig ousterConfig(int height) {
  auto config = smallConfig();
  config.projection = "ouster_lut";
  config.width = 4;
  config.height = height;
  config.window_width = 1;
  config.pixel_shift_by_row.resize(height, 0);
  return config;
}

void ousterLookupPreservesOriginalPointIndex() {
  auto cloud = cloudOf(8);
  cloud.scan_width = 4;
  cloud.scan_height = 2;
  cloud.intensities().setConstant(-1.0);
  cloud.intensities()(0, 1) = 10.0;
  cloud.intensities()(0, 4) = 30.0;
  auto config = ousterConfig(2);
  config.pixel_shift_by_row = {0, 1};
  const auto frame = bievr::normalizeIntensity(cloud, config, 0.1, 100.0);
  near(frame.values(1), 140.0 * 10.0 / 21.0, "Ouster first row was projected incorrectly");
  near(frame.values(4), 140.0 * 30.0 / 21.0, "Ouster shift direction or row index incorrect");
  require(frame.valid[0] == 0 && frame.valid[1] == 1 && frame.valid[4] == 1,
          "Ouster lookup reordered point validity");
}

void ousterWrongOrganizationIsRejected() {
  auto cloud = cloudOf(8);
  cloud.scan_width = 8;
  cloud.scan_height = 1;
  bool rejected = false;
  try { bievr::normalizeIntensity(cloud, ousterConfig(2), 0.1, 100.0); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "unorganized Ouster cloud silently used a wrong lookup");
}

void ousterLineArtifactIsRemovedBeforeNormalization() {
  auto cloud = cloudOf(12);
  cloud.scan_width = 4;
  cloud.scan_height = 3;
  cloud.intensities() << 10, 10, 10, 10, 20, 20, 20, 20, 10, 10, 10, 10;
  auto config = ousterConfig(3);
  config.window_height = 1;
  config.line_removal = true;
  config.highpass = {-0.5, 1.0, -0.5};
  config.lowpass = {1.0};
  const auto frame = bievr::normalizeIntensity(cloud, config, 0.1, 100.0);
  near(frame.values(0), 140.0 * 20.0 / 21.0, "vertical filter border handling wrong");
  near(frame.values(4), 140.0 * 10.0 / 11.0, "row artifact not removed");
  cloud.intensities()(0, 0) = -1.0;
  const auto sparse = bievr::normalizeIntensity(cloud, config, 0.1, 100.0);
  near(sparse.values(4), 140.0 * 20.0 / 21.0,
       "missing pixel manufactured a line correction");
}

void verticalBoundariesDoNotWrap() {
  auto cloud = cloudOf(2);
  cloud.data().col(0).head<3>() << 0.0, 0.0, 2.0;
  cloud.data().col(1).head<3>() << 0.0, 0.0, -2.0;
  cloud.intensities() << 10.0, 30.0;
  const auto frame = bievr::normalizeIntensity(cloud, smallConfig(), 0.1, 100.0);
  require(frame.valid == std::vector<uint8_t>({1, 1}), "exact polar return lost");
  near(frame.values(0), 140.0 * 10.0 / 11.0, "vertical image top wrapped");
  near(frame.values(1), 140.0 * 30.0 / 31.0, "vertical image bottom wrapped");
  auto config = smallConfig();
  config.vertical_fov_deg = 90.0;
  const auto cropped = bievr::normalizeIntensity(cloud, config, 0.1, 100.0);
  require(cropped.valid == std::vector<uint8_t>({0, 0}), "out-of-FOV return was projected");
}

void configurationCanBeValidatedBeforeReceivingScans() {
  auto config = ousterConfig(2);
  config.line_removal = true;
  config.highpass = {1.0, -1.0};
  config.lowpass = {1.0};
  bool rejected = false;
  try { bievr::validateIntensityConfig(config); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "even line filter accepted before receiving a scan");

  config.highpass = {std::numeric_limits<double>::quiet_NaN()};
  rejected = false;
  try { bievr::validateIntensityConfig(config); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "nonfinite line filter accepted before receiving a scan");

  config = smallConfig();
  config.width = 32769;
  rejected = false;
  try { bievr::validateIntensityConfig(config); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "unsupported image dimensions accepted before receiving a scan");

  config.enabled = false;
  bievr::validateIntensityConfig(config);
  bievr::validateIntensityConfig(smallConfig());
}

}  // namespace

int main() {
  try {
    sparseBrightnessIgnoresEmptyPixels();
    collidingPixelsPreserveIndividualReturns();
    zeroIntensityRemainsAnObservedPixel();
    invalidReturnsDoNotContaminateNeighborhoods();
    sphericalNeighborhoodWrapsAcrossAzimuthSeam();
    rawScalingAndSaturation();
    emptyAndDisabledClouds();
    invalidConfigurationIsRejected();
    absentIntensityDisablesOnlyTheIntensityChannel();
    ousterLookupPreservesOriginalPointIndex();
    ousterWrongOrganizationIsRejected();
    ousterLineArtifactIsRemovedBeforeNormalization();
    verticalBoundariesDoNotWrap();
    configurationCanBeValidatedBeforeReceivingScans();
    std::cout << "Intensity normalization tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << "Intensity normalization test failed: " << error.what() << '\n';
    return 1;
  }
}
