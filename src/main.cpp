#include <future>
#include <iostream>
#include <vector>

#include "cut_plane.hpp"
#include "diversity_problem.hpp"
#include "glover.hpp"
#include "partitioned_solver.hpp"
#include "semaphore.hpp"

using std::vector;

void run_test(string type, int n, int p, int s, int k,
              const vector<int> timelimits, const vector<string> &strategies,
              const vector<double> &partition_ratios, const string &filename,
              Semaphore *sem, bool ct, bool all, bool glover) {
  // Create a DiversityProblem instance with random locations
  const DiversityProblem dp(type, n, p, s, k);
  bool low_mem = false;
  // Solve under each strategy
  for (const string &strategy : strategies) {
    for (const double ratio : partition_ratios) {
      CoordinatePartitionSolver cps(dp, strategy, ratio, low_mem);
      cps.set_timelimit(timelimits);
      cps.solve();
      cps.write_results(filename);
    }
  }
  // Solve on cut-plane
  if (ct) {
    CutPlaneSolver ct(dp, low_mem);
    ct.set_timelimit(timelimits);
    ct.solve();
    ct.write_results(filename);
  }

  // Solve fully partitioned
  if (all) {
    CoordinatePartitionSolver cps_full(dp, "all", 1.0, low_mem);
    cps_full.set_timelimit(timelimits);
    cps_full.solve();
    cps_full.write_results(filename);
  }

  // Solve on glover
  if (glover) {
    GloverSolver gv(dp);
    gv.set_timelimit(timelimits);
    gv.solve();
    gv.write_results(filename);
  }

  // Finish up
  sem->notify();
}

void run_parallel_test(int threads, const string type, const vector<int> &N,
                       const vector<double> &P_ratio, const vector<int> &S,
                       const int K, const vector<int> &timelimits,
                       const vector<string> &strategies,
                       const vector<double> &partition_ratios,
                       const string filename, bool ct, bool all, bool glover) {
  // Create a vector to store our futures objects
  Semaphore sem(threads);
  vector<std::future<void>> futures;
  for (int n : N) {
    for (double p_ratio : P_ratio) {
      int p = n * p_ratio;
      for (int s : S) {
        for (int k = 0; k < K; k++) {
          sem.wait();
          futures.push_back(std::async(
              std::launch::async, run_test, type, n, p, s, k, timelimits,
              strategies, partition_ratios, filename, &sem, ct, all, glover));
        }
      }
    }
  }
  // Wait for all threads to complete
  for (auto &future : futures) {
    future.wait();
  }
}

void cube_test() {
  // Solver settings
  const vector<int> timelimits = {30, 60, 120, 300, 600, 1000};
  const vector<string> strategies = {"random", "stratified", "greedy",
                                     "stepped"};
  const vector<double> partition_ratios = {0.75, 0.5, 0.25, 0.1};
  const bool ct = true;
  const bool all = true;
  const bool glover = false;
  // Problem settings
  const vector<int> N = {1000};
  const vector<double> P_ratio = {0.1, 0.2};
  const vector<int> S = {20};
  const int K = 5;
  // Output
  const string filename = "data/cube.csv";
  // Run parallel
  run_parallel_test(1, "random", N, P_ratio, S, K, timelimits, strategies,
                    partition_ratios, filename, ct, all, glover);
}

void ball_test() {
  // Solver settings
  const vector<int> timelimits = {30, 60, 120, 300, 600, 1000};
  const vector<string> strategies = {"random", "stratified"};
  const vector<double> partition_ratios = {0.25, 0.5, 0.75};
  const bool ct = true;
  const bool all = true;
  const bool glover = true;
  // Problem settings
  const vector<int> N = {10, 25, 50, 100};
  const vector<double> P_ratio = {0.1, 0.2};
  const vector<int> S = {5, 2};
  const int K = 5;
  // Output
  const string filename = "data/ball.csv";
  // Run parallel
  run_parallel_test(1, "circle", N, P_ratio, S, K, timelimits, strategies,
                    partition_ratios, filename, ct, all, glover);
}

int main() {
  cube_test();
  // ball_test();
  return 0;
}
