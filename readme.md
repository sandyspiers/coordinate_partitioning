# Coordinate Partitioning for Difficult Euclidean Max-Sum Diversity Problems

This repo contains a new exact method to solve difficult and high-coordinate Euclidean max-sum diversity problems.
It is associated with [1], and comes as a result of our work in [2].

## Dependencies

* `Cplex 22.1.1`
* `Eigen 3.4.0`

## Reproducibility

To reproduce the results in [1], update `CMakeLists.txt` to include corrected filepaths for `Cplex` and `Eigen` libraries and run

```bash
mkdir build
cd build
cmake ..
make .
```

Then to run the experiments in the background, run

```bash
nohup ./build/coordinate_partitioning &
```

## Contents

The repo is currently a work in progress, more documentation to come soon.

## References

[1] Spiers, S., Bui, H. T., & Loxton, R. (2023). Coordinate Partitioning for Difficult Euclidean Max-Sum Diversity Problems. *Manuscript under review*.

[2] [Spiers, S., Bui, H. T., & Loxton, R. (2023). An exact cutting plane method for the Euclidean max-sum diversity problem. *European Journal of Operational Research*.](https://www.sciencedirect.com/science/article/pii/S037722172300379X).
