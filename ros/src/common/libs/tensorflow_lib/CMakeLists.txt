cmake_minimum_required(VERSION 2.8.12)
project(tensorflow_lib)

find_package(catkin REQUIRED COMPONENTS)

find_package(CUDA)

SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

find_library(TENSORFLOW tensorflow)
if (TENSORFLOW)

  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES tensorflow_lib
    )
  include_directories(tensorflow_lib
    include
    ${catkin_INCLUDE_DIRS}
    )
  add_library(tensorflow_lib
    src/tensorflow_lib.cpp
    )
  target_link_libraries(tensorflow_lib
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
  message("'TensorFlow' is not installed. The library 'tensorflow_lib' will not be built.")
endif ()
