#ifndef COORD_PAR_SOLVER_H
#define COORD_PAR_SOLVER_H

#include <ilcplex/ilocplex.h>

#include <functional>

#include "partitioned_instance.hpp"
#include "solver.hpp"

using std::function;

class DiversityProblem;

class CoordinatePartitionSolver : public Solver {
 private:
  // Cplex stuff
  IloEnv env;
  IloModel model;
  IloCplex cplex;

  // Model dvars
  IloBoolVarArray x;
  IloNumVarArray theta;

  // Name override
  int num_cuts = 0;
  string solver_name = "coordpar";

  // Full EDM (if required).
  // Should be lower triangular
  vector<vector<vector<double>>> edm;

  // Distance function, should be explicitly instantiated based on strategy.
  // Either recalls the edm, or generates explicitly
  // get_dist(partition, i, j)
  function<const double(int, int, int)> get_dist;

  void setup_model();

 public:
  const PartitionedInstance pi;
  // Constructor
  CoordinatePartitionSolver(const DiversityProblem &dp, string strategy,
                            bool low_memory = true);
  CoordinatePartitionSolver(const DiversityProblem &dp, string strategy,
                            int num_partitions, bool low_memory = true);
  CoordinatePartitionSolver(const DiversityProblem &dp, string strategy,
                            double partition_ratio, bool low_memory = true);

  // Solve parameters
  const bool low_memory_cuts;

  // Destructor
  ~CoordinatePartitionSolver() {
    cplex.end();
    env.end();
  }

  void solve() override;
  Result get_result(int timelimit, double solve_time) const;
  Result get_result(double timelimit, double obj_val, double best_bound,
                    double gap, double solve_time) const override;
};

#endif
