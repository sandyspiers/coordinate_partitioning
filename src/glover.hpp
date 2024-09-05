#ifndef GLOVER_H
#define GLOVER_H

#include <ilcplex/ilocplex.h>
#include <functional>

#include "solver.hpp"

using std::function;

class GloverSolver : public Solver {
 private:
  // Cplex stuff
  IloEnv env;
  IloModel model;
  IloCplex cplex;

  // Model dvars
  IloBoolVarArray x;
  IloNumVarArray w;

  // Info
  string solver_name = "glover";

  // Full EDM
  // Is built to be lower triangular!!!!
  // Therefore edm[i][j] exists only if i<j.
  // if i==j, then process this out as 0
  // if i > j, just call edm[j][i].
  vector<vector<double>> edm;

 public:
  // Constructor and model builder
  GloverSolver(const DiversityProblem& problem);

  // Destructor
  ~GloverSolver() {
    cplex.end();
    env.end();
  }

  void solve() override;
  Result get_result(int timelimit, double solve_time) const;
  Result get_result(double timelimit, double obj_val, double best_bound,
                    double gap, double solve_time) const override;
};

#endif
