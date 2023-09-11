#include <future>
#include <iostream>
#include <vector>

#include "cut_plane.hpp"
#include "diversity_problem.hpp"
#include "partitioned_solver.hpp"
#include "semaphore.hpp"

using std::vector;

void run_test(string type, int n, int p, int s, int k,
              const vector<int> timelimits, const vector<string>& strategies,
              const vector<double>& partition_ratios, const string& filename,
              Semaphore* sem) {
  // Create a DiversityProblem instance with random locations
  DiversityProblem dp(type, n, p, s, k);
  bool low_mem = false;
  // Solve under each strategy
  for (const string& strategy : strategies) {
    for (const double ratio : partition_ratios) {
      // low_mem = (ratio >= 0.49);
      CoordinatePartitionSolver cps(dp, strategy, ratio, low_mem);
      cps.set_timelimit(timelimits);
      cps.solve();
      cps.write_results(filename);
    }
  }
  // Solve on cut-plane
  CutPlaneSolver ct(dp, false);
  ct.set_timelimit(timelimits);
  ct.solve();
  ct.write_results(filename);

  // Finish up
  sem->notify();
}

void random_box_test(const vector<int> timelimits,
                     const vector<string>& strategies,
                     const vector<double>& partition_ratios) {
  // Problem settings
  const vector<int> N = {500, 250, 100};
  const vector<double> P_ratio = {0.1, 0.2};
  const vector<int> S = {20, 15, 10, 5, 2};
  const int K = 5;
  // Output
  const string filename = "results/random_box.txt";

  // Create a vector to store our futures objects
  Semaphore sem(K * 2);
  vector<std::future<void>> futures;
  for (int n : N) {
    for (double p_ratio : P_ratio) {
      int p = n * p_ratio;
      for (int s : S) {
        for (int k = 0; k < K; k++) {
          sem.wait();
          futures.push_back(std::async(std::launch::async, run_test, "random",
                                       n, p, s, k, timelimits, strategies,
                                       partition_ratios, filename, &sem));
        }
      }
    }
  }
  // Wait for all threads to complete
  for (auto& future : futures) {
    future.wait();
  }
}

void random_box_test_large(const vector<int> timelimits,
                           const vector<string>& strategies,
                           const vector<double>& partition_ratios) {
  // Problem settings
  const vector<int> N = {1000};
  const vector<double> P_ratio = {0.1, 0.2};
  const vector<int> S = {20, 15, 10, 5, 2};
  const int K = 5;
  // Output
  const string filename = "results/random_box_large.txt";

  // Create a vector to store our futures objects
  Semaphore sem(K * 2);
  vector<std::future<void>> futures;
  for (int n : N) {
    for (double p_ratio : P_ratio) {
      int p = n * p_ratio;
      for (int s : S) {
        for (int k = 0; k < K; k++) {
          sem.wait();
          futures.push_back(std::async(std::launch::async, run_test, "random",
                                       n, p, s, k, timelimits, strategies,
                                       partition_ratios, filename, &sem));
        }
      }
    }
  }
  // Wait for all threads to complete
  for (auto& future : futures) {
    future.wait();
  }
}

void random_circle_test(const vector<int> timelimits,
                        const vector<string>& strategies,
                        const vector<double>& partition_ratios) {
  // Problem settings
  const vector<int> N = {250, 100, 50};
  const vector<double> P_ratio = {0.1, 0.2};
  const vector<int> S = {10, 5, 2};
  const int K = 5;
  // Output
  const string filename = "results/random_circle.txt";

  // Create a vector to store our futures objects
  Semaphore sem(K * 2);
  vector<std::future<void>> futures;
  for (int n : N) {
    for (double p_ratio : P_ratio) {
      int p = n * p_ratio;
      for (int s : S) {
        for (int k = 0; k < K; k++) {
          sem.wait();
          futures.push_back(std::async(std::launch::async, run_test, "circle",
                                       n, p, s, k, timelimits, strategies,
                                       partition_ratios, filename, &sem));
        }
      }
    }
  }
  // Wait for all threads to complete
  for (auto& future : futures) {
    future.wait();
  }
}

int main() {
  // Solver settings
  const vector<int> timelimits = {30, 60, 120, 300, 600};
  const vector<string> strategies = {"random", "stratified", "greedy",
                                     "stepped"};
  const vector<double> partition_ratios = {0.75, 0.5, 0.25, 0.1, 0.05, 0.01};

  random_box_test(timelimits, strategies, partition_ratios);
  random_box_test_large(timelimits, strategies, partition_ratios);
  random_circle_test(timelimits, strategies, partition_ratios);

  return 0;
}
