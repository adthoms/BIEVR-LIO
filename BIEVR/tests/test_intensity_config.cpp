#include "bievr_lio/config_loader.h"

#include <cstdlib>
#include <iostream>

bool intensityEnabled(const bievr::Pipeline::Config& config) {
  return config.intensity.enabled;
}

int main(int argc, char** argv) {
  if (argc != 2) return EXIT_FAILURE;
  const std::string root = argv[1];
  bievr::Config config;
  if (!bievr::loadConfigFromYaml({root + "/config/params.yaml",
                                 root + "/config/sensor_configs/enwide.yaml"}, config)) {
    return EXIT_FAILURE;
  }
  if (!intensityEnabled(config.pipeline_config)) {
    std::cerr << "COIN-BIEVR must be enabled by default\n";
    return EXIT_FAILURE;
  }
  const auto paths = std::vector<std::string>{root + "/config/params.yaml",
                                              root + "/config/sensor_configs/enwide.yaml"};
  auto disabled_paths = paths;
  disabled_paths.push_back(root + "/BIEVR/tests/data/intensity_disabled.yaml");
  if (!bievr::loadConfigFromYaml(disabled_paths, config) || intensityEnabled(config.pipeline_config)) {
    std::cerr << "disabled intensity must not require profile files\n";
    return EXIT_FAILURE;
  }
  auto invalid_paths = paths;
  invalid_paths.push_back(root + "/BIEVR/tests/data/intensity_invalid.yaml");
  if (bievr::loadConfigFromYaml(invalid_paths, config)) {
    std::cerr << "even intensity windows must fail configuration loading\n";
    return EXIT_FAILURE;
  }
  auto override_paths = paths;
  override_paths.push_back(root + "/BIEVR/tests/data/intensity_override.yaml");
  if (!bievr::loadConfigFromYaml(override_paths, config) ||
      config.pipeline_config.intensity.raw_scale != 2.0 ||
      config.pipeline_config.registration.photo_scale != 0.002) {
    std::cerr << "explicit configuration must override profile defaults\n";
    return EXIT_FAILURE;
  }
  std::cout << "intensity config passed\n";
}
