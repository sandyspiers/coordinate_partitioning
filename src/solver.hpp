#ifndef SOLVER_H
#define SOLVER_H

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "result.hpp"

using std::string;
using std::vector;

class DiversityProblem;

class Solver {
 protected:
  // Timelimits
  vector<int> timelimits;
  // Any intermediate solve results (used if multiple timelimits)
  vector<Result> results;
  // Solution
  vector<int> solution;
  vector<int> nodes;
  // Metrics
  string solver_name = "solver";
  std::chrono::_V2::system_clock::time_point start_time;
  double setup_time = 0.0;

 public:
  // Constructors should be defined by the derived class,
  // However all should be something like
  Solver(const DiversityProblem& problem) : dp(problem) {
    start_time = std::chrono::high_resolution_clock::now();
  };

  // Virtual destructor
  // virtual ~Solver() {}

  // Problem instance
  const DiversityProblem& dp;

  // Set timelimits
  void set_timelimit(int timelimit) {
    timelimits.push_back(timelimit);
    std::sort(timelimits.begin(), timelimits.end());
  }
  void set_timelimit(vector<int> timelimits) {
    this->timelimits = timelimits;
    std::sort(timelimits.begin(), timelimits.end());
  }
  void set_timelimits(vector<int> timelimits) { set_timelimit(timelimits); }

  // Solve function
  virtual void solve() = 0;
  virtual Result get_result(double timelimit, double obj_val, double gap,
                            double best_bound, double solve_time) const = 0;

  // Getters
  const Result& solve_details() const { return results.back(); }
  const vector<Result>& get_results() const { return results; }
  const vector<int>& get_sol() const { return solution; }
  const vector<int>& get_nodes() const { return nodes; }
  const double get_setup_time() const { return setup_time; }
  const string& get_name() const { return solver_name; }
  void write_results(string filename) const {
    for (const Result& r : results) r.write(filename);
  };
};

#endif
