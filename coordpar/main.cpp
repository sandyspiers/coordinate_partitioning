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
              const vector<int> timelimits, const vector<string>& strategies,
              const vector<double>& partition_ratios, const string& filename,
              Semaphore* sem, bool glover) {
  // Create a DiversityProblem instance with random locations
  const DiversityProblem dp(type, n, p, s, k);
  bool low_mem = false;
  // Solve under each strategy
  for (const string& strategy : strategies) {
    for (const double ratio : partition_ratios) {
      CoordinatePartitionSolver cps(dp, strategy, ratio, low_mem);
      cps.set_timelimit(timelimits);
      cps.solve();
      cps.write_results(filename);
    }
  }
  // Solve on cut-plane
  CutPlaneSolver ct(dp, low_mem);
  ct.set_timelimit(timelimits);
  ct.solve();
  ct.write_results(filename);

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

void run_parallel_test(int threads, const vector<int>& N,
                       const vector<double>& P_ratio, const vector<int>& S,
                       const int K, const vector<int>& timelimits,
                       const vector<string>& strategies,
                       const vector<double>& partition_ratios,
                       const string filename, bool glover) {
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
              std::launch::async, run_test, "random", n, p, s, k, timelimits,
              strategies, partition_ratios, filename, &sem, glover));
        }
      }
    }
  }
  // Wait for all threads to complete
  for (auto& future : futures) {
    future.wait();
  }
}

void random_box_test_small() {
  // Solver settings
  const vector<int> timelimits = {30, 60, 120, 300, 600};
  const vector<string> strategies = {"random", "stratified", "greedy",
                                     "stepped"};
  const vector<double> partition_ratios = {0.75, 0.5, 0.25, 0.1};
  const bool glover = false;
  // Problem settings
  const vector<int> N = {500, 250, 100};
  const vector<double> P_ratio = {0.1, 0.2};
  const vector<int> S = {20, 15, 10, 5, 2};
  const int K = 5;
  // Output
  const string filename = "res/random_box_small.txt";
  // Run parallel, using all cores
  run_parallel_test(16, N, P_ratio, S, K, timelimits, strategies,
                    partition_ratios, filename, glover);
}

void random_box_test_large() {
  // Solver settings
  const vector<int> timelimits = {30, 60, 120, 300, 600};
  const vector<string> strategies = {"random", "stratified", "greedy",
                                     "stepped"};
  const vector<double> partition_ratios = {0.75, 0.5, 0.25, 0.1};
  const bool glover = false;
  // Problem settings
  const vector<int> N = {1000};
  const vector<double> P_ratio = {0.1, 0.2};
  const vector<int> S = {20, 15, 10, 5, 2};
  const int K = 5;
  // Output
  const string filename = "res/random_box_large.txt";
  // Run parallel, using half cores
  run_parallel_test(8, N, P_ratio, S, K, timelimits, strategies,
                    partition_ratios, filename, glover);
}

void random_circle_test() {
  // Solver settings
  const vector<int> timelimits = {30, 60, 120, 300, 600};
  const vector<string> strategies = {"random", "stratified"};
  const vector<double> partition_ratios = {0.75, 0.5, 0.25};
  const bool glover = true;
  // Problem settings
  const vector<int> N = {100, 50};
  const vector<double> P_ratio = {0.1, 0.2};
  const vector<int> S = {5, 2};
  const int K = 5;
  // Output
  const string filename = "res/random_circle.txt";
  // Run parallel, using all cores
  run_parallel_test(16, N, P_ratio, S, K, timelimits, strategies,
                    partition_ratios, filename, glover);
}

int main() {
  random_box_test_small();
  random_box_test_large();
  random_circle_test();
  return 0;
}
