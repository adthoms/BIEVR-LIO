#include "bievr_ros_common/conversions.h"

#include <cstdlib>
#include <iostream>

namespace {
void check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

bool hasIntensity(const bievr::StampedIntensityPointcloud& cloud) {
  return cloud.has_intensity;
}

sensor_msgs::msg::PointCloud2 cloudMessage(bool intensity, uint8_t datatype) {
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.stamp.sec = 1;
  msg.height = 1;
  msg.width = 2;
  msg.point_step = 20;
  msg.row_step = 40;
  msg.data.resize(40);
  const char* names[] = {"x", "y", "z", "t", "intensity"};
  for (int i = 0; i < (intensity ? 5 : 4); ++i) {
    sensor_msgs::msg::PointField field;
    field.name = names[i];
    field.offset = 4 * i;
    field.count = 1;
    field.datatype = i == 3 ? sensor_msgs::msg::PointField::UINT32
                           : (i == 4 ? datatype : sensor_msgs::msg::PointField::FLOAT32);
    msg.fields.push_back(field);
  }
  for (size_t i = 0; i < 2; ++i) {
    const float x = 2.0f + i;
    std::memcpy(msg.data.data() + 20 * i, &x, sizeof(x));
  }
  return msg;
}
}  // namespace

int main() {
  using Field = sensor_msgs::msg::PointField;
  bievr::StampedIntensityPointcloud cloud;
  auto missing = cloudMessage(false, Field::FLOAT32);
  check(bievr::msgToPointcloud(missing, cloud), "missing intensity must retain geometry");
  check(!hasIntensity(cloud), "missing intensity must be distinguishable from a valid zero");
  auto zero = cloudMessage(true, Field::FLOAT32);
  check(bievr::msgToPointcloud(zero, cloud) && hasIntensity(cloud), "zero intensity is valid");
  auto integer = cloudMessage(true, Field::UINT16);
  const uint16_t value = 1234;
  std::memcpy(integer.data.data() + 16, &value, sizeof(value));
  check(bievr::msgToPointcloud(integer, cloud), "uint16 intensity conversion failed");
  check(cloud.intensities()(0, 0) == value, "integer intensity must preserve its value");
  auto padded = cloudMessage(true, Field::FLOAT32);
  padded.height = 2;
  padded.row_step = 48;
  padded.data.resize(96);
  const float x = 7.0f;
  const float intensity = 45.0f;
  std::memcpy(padded.data.data() + 48, &x, sizeof(x));
  std::memcpy(padded.data.data() + 48 + 16, &intensity, sizeof(intensity));
  check(bievr::msgToPointcloud(padded, cloud), "organized cloud conversion failed");
  check(cloud[2].x() == x && cloud.intensities()(0, 2) == intensity,
        "organized clouds must respect row padding and intensity alignment");
  check(cloud.scan_width == 2 && cloud.scan_height == 2, "original layout must be retained");
  std::cout << "intensity conversion passed\n";
}
