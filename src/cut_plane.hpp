#ifndef CUT_PLANE_H
#define CUT_PLANE_H

#include <ilcplex/ilocplex.h>

#include <functional>

#include "solver.hpp"

using std::function;

class CutPlaneSolver : public Solver {
private:
  // Cplex stuff
  IloEnv env;
  IloModel model;
  IloCplex cplex;

  // Model dvars
  IloBoolVarArray x;
  IloNumVar theta;

  // Info
  int num_cuts = 0;
  string solver_name = "ct";

  // Full EDM (if required)
  // Is built to be lower triangular!!!!
  // Therefore edm[i][j] exists only if i<j.
  // if i==j, then process this out as 0
  // if i > j, just call edm[j][i].
  vector<vector<double>> edm;

  // Distance function, should be explicitly instantiated based on strategy.
  // Either recalls the edm, or generates explicitly
  function<const double(int, int)> get_dist;

public:
  // Constructor and model builder
  CutPlaneSolver(const DiversityProblem &problem, const bool low_memory_cuts);

  // Solver settings
  const bool low_memory_cuts;

  // Destructor
  ~CutPlaneSolver() {
    cplex.end();
    env.end();
  }

  void solve() override;
  Result get_result(int timelimit, double solve_time) const;
  Result get_result(double timelimit, double obj_val, double best_bound,
                    double gap, double solve_time) const override;
};

#endif
