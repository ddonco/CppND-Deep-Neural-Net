# This file exports the Eigen3::Eigen CMake target which should be passed to the
# target_link_libraries command.

@PACKAGE_INIT@

include ("${CMAKE_CURRENT_LIST_DIR}/Eigen3Targets.cmake")

# Legacy variables, do *not* use. May be removed in the future.

set (EIGEN3_FOUND 1)
set (EIGEN3_USE_FILE    "${CMAKE_CURRENT_LIST_DIR}/UseEigen3.cmake")

set (EIGEN3_DEFINITIONS  "@EIGEN_DEFINITIONS@")
set (EIGEN3_INCLUDE_DIR  "@PACKAGE_EIGEN_INCLUDE_DIR@")
set (EIGEN3_INCLUDE_DIRS "@PACKAGE_EIGEN_INCLUDE_DIR@")
set (EIGEN3_ROOT_DIR     "@PACKAGE_EIGEN_ROOT_DIR@")

set (EIGEN3_VERSION_STRING "@EIGEN_VERSION_STRING@")
set (EIGEN3_VERSION_MAJOR  "@EIGEN_VERSION_MAJOR@")
set (EIGEN3_VERSION_MINOR  "@EIGEN_VERSION_MINOR@")
set (EIGEN3_VERSION_PATCH  "@EIGEN_VERSION_PATCH@")

message("eigen verson : ${EIGEN3_VERSION_STRING}")
# set(EIGEN3_VERSION 3.3)