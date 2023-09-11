cmake_minimum_required(VERSION 3.0.0)
project(cordpar VERSION 0.1.0 LANGUAGES C CXX)

project(coordinate_partition)

# Eigen directories
set(EIGEN_INCLUDE_DIR "/home/ubuntu/lib/eigen-3.4.0")

# CPLEX directories
set(CPLEX_ROOT_DIR "/home/ubuntu/opt/ibm/ILOG/CPLEX_Studio2211")
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
add_executable(coordpar 
coordpar/main.cpp 
coordpar/result.cpp
coordpar/diversity_problem.cpp 
coordpar/cut_plane.cpp
coordpar/partitioned_instance.cpp
coordpar/partitioned_solver.cpp
)
target_link_libraries(coordpar ilocplex concert cplex m pthread)
