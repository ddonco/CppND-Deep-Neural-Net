cmake_minimum_required(VERSION 3.10)

add_definitions(-std=c++17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CXX_FLAGS "-Wall") # -march=native -fopenmp")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(DNN)

# Find all executables
file(GLOB project_SRCS src/*.cpp) #src/*.h

add_executable(DNN ${project_SRCS})

# Find Eigen
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find Matplotlib
find_package(Python3 COMPONENTS Development NumPy)
target_include_directories(DNN PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(DNN Python3::Python Python3::NumPy)

message("${CMAKE_CURRENT_LIST_DIR}")
message("${CMAKE_MODULE_PATH}")
message("${EIGEN3_INCLUDE_DIR}")
message("${Python3_INCLUDE_DIRS}")
