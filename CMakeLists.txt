cmake_minimum_required(VERSION 2.8.3)
project(radar_visualization)

find_package(catkin REQUIRED COMPONENTS
  cacc_msgs
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
)

catkin_package( )

#############
## Install ##
#############
install(
  DIRECTORY
    scripts
  USE_SOURCE_PERMISSIONS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)