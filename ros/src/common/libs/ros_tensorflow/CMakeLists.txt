cmake_minimum_required(VERSION 2.8.12)
project(ros_tensorflow)

find_package(catkin REQUIRED COMPONENTS)

find_package(CUDA)

SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

find_library(TENSORFLOW tensorflow)
if (TENSORFLOW)

  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ros_tensorflow
    )
  include_directories(ros_tensorflow
    include
    ${catkin_INCLUDE_DIRS}
    )
  add_library(ros_tensorflow
    src/ros_tensorflow.cpp
    )
  target_link_libraries(ros_tensorflow
    ${catkin_LIBRARIES}
    )
  add_dependencies(ros_tensorflow
    ${catkin_EXPORTED_TARGETS}
    )
  install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp"
    )
  install(TARGETS ros_tensorflow
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
else ()
  message("'TensorFlow' is not installed. The library 'ros_tensorflow' will not be built.")
endif ()
