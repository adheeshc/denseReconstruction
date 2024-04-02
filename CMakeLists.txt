cmake_minimum_required(VERSION 3.0.0)
project(projectName)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Ceres REQUIRED)
find_package(g2o REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})

include_directories(${CERES_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})

set(source_files
    src/denseMapping.cpp
)

add_executable(monoDepth ${source_files})

target_link_libraries(monoDepth ${OpenCV_LIBS})
target_link_libraries(monoDepth Eigen3::Eigen)
target_link_libraries(monoDepth Sophus::Sophus)
target_link_libraries(monoDepth ${Pangolin_LIBRARIES})
target_link_libraries(monoDepth Ceres::ceres)
target_link_libraries(monoDepth ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CSPARSE} ${G2O_TYPES_SLAM2D} ${G2O_TYPES_SLAM3D} ${G2O_TYPES_SBA})
target_link_libraries(monoDepth g2o_core g2o_stuff g2o_solver_csparse g2o_solver_dense g2o_solver_eigen g2o_types_sba g2o_types_slam2d g2o_types_slam3d g2o_viewer)

# Add the -pthread flag to the compiler and linker flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
