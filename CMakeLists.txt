cmake_minimum_required(VERSION 3.10)
project(ReadPCD CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(read_pcd /ist-nas/users/palakonk/singularity/home/palakons/linear_hough_man_radar/main.cpp)
target_link_libraries(read_pcd ${PCL_LIBRARIES})