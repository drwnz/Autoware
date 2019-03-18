cmake_minimum_required(VERSION 2.8.12)
project(tensorflow_lib)

FIND_PACKAGE(catkin REQUIRED COMPONENTS)

FIND_PACKAGE(CUDA)
FIND_PACKAGE(OpenCV REQUIRED)


SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

find_library(TENSORFLOW tensorflow)
if (TENSORFLOW)

  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES tensorflow_lib
    )
  INCLUDE_DIRECTORIES(tensorflow_lib
    include
    ${catkin_INCLUDE_DIRS}
    ${autoware_msgs_INCLUDE_DIRS}
    )
  ADD_LIBRARY(tensorflow_lib
    src/tensorflow_lib.cpp
    )
  TARGET_LINK_LIBRARIES(tensorflow_lib
    ${catkin_LIBRARIES}
    )
  add_dependencies(tensorflow_lib
    ${catkin_EXPORTED_TARGETS}
    )
  install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp"
    )
  install(TARGETS tensorflow_lib
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
else ()
  message("'Tensorflow' is not installed. The library 'tensorflow_lib' will not be built.")
endif ()
