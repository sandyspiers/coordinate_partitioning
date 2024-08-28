cmake_minimum_required(VERSION 3.0.0)
project(cordpar VERSION 0.1.0 LANGUAGES C CXX)

project(coordinate_partition)

# Include dir
set(INCLUDE_DIR "/usr/include/")

# Eigen directories
set(EIGEN_INCLUDE_DIR "${INCLUDE_DIR}/eigen3")

# CPLEX directories
set(CPLEX_ROOT_DIR "/opt/ibm/ILOG/CPLEX_Studio2211")
set(CPLEX_INCLUDE_DIR "${CPLEX_ROOT_DIR}/cplex/include")
set(CPLEX_LIB_DIR "${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic")
set(CONCERT_INCLUDE_DIR "${CPLEX_ROOT_DIR}/concert/include")
set(CONCERT_LIB_DIR "${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic")

# Include directories
include_directories(${CPLEX_INCLUDE_DIR})
include_directories(${CONCERT_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIR})

# Link directories
link_directories(${CPLEX_LIB_DIR})
link_directories(${CONCERT_LIB_DIR})

# Optimisation
add_compile_options(-O3)

# Main solver
add_executable(coordinate_partitioning 
    src/main.cpp 
    src/result.cpp
    src/diversity_problem.cpp 
    src/cut_plane.cpp
    src/partitioned_instance.cpp
    src/partitioned_solver.cpp
    src/glover.cpp
)
target_link_libraries(coordinate_partitioning ilocplex concert cplex m pthread)
