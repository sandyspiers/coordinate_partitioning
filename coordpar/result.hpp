#ifndef RESULT_H
#define RESULT_H

#include <ilcplex/ilocplex.h>

#include <string>
#include <vector>

class Solver;

using std::string;
using std::vector;

/*
A class to hold results for the purpose of comparison between solvers.
I need things like
 - problem:
    - name(seed)
    - nodes
    - coords
    - cardinality
 - solver:
    - solver_name
    - strategy
    - num_coords(partition only)
    - num_partitions(partition only)
 - result:
    - timelimit
    - num_cuts
    - obj_val
    - best_bound
    - setup_time
    - solve_time
    - total_time
 */
class Result {
 public:
  // Problem related
  const string problem_name;
  const int num_nodes;
  const int num_coords;
  const int cardinality;
  // Solver related
  const string solver_name;
  const string strategy = "na";
  const int num_recovered_coords = -1;
  const int num_partitions = -1;
  // Results
  const int timelimit;
  const int num_cuts;
  const double obj_val;
  const double best_bound;
  const double gap;
  const double setup_time;
  const double solve_time;
  const double total_time;
  // All explicit
  Result(const string& problem_name, int num_nodes, int num_coords,
         int cardinality, const string& solver_name, const string& strategy,
         int num_recovered_coords, int num_partitions, int timelimit,
         int num_cuts, double obj_val, double best_bound, double gap,
         double setup_time, double solve_time, double total_time);
  string get_string() const;
  void write(string filename) const;
};

class IntermediateResultsI : public IloCplex::MIPInfoCallbackI {
  // Per result
  Solver& solver;
  IloNum& start_time;
  // Tracking
  vector<int>& timelimits;
  vector<Result>& results;

 public:
  IloCplex::CallbackI* duplicateCallback() const override {
    return (new (getEnv()) IntermediateResultsI(*this));
  }
  IntermediateResultsI(IloEnv env, IloNum& start_time, vector<int>& timelimits,
                       vector<Result>& results, Solver& solver)
      : IloCplex::MIPInfoCallbackI(env),
        start_time(start_time),
        timelimits(timelimits),
        results(results),
        solver(solver) {}
  void main() override;
};
IloCplex::Callback IntermediateResults(IloEnv env, IloNum& start_time,
                                       vector<int>& timelimits,
                                       vector<Result>& results, Solver& solver);

#endif