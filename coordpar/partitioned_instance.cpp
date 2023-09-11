#include "partitioned_instance.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "diversity_problem.hpp"

using std::cerr;
using std::cout;
using std::endl;

PartitionedInstance::PartitionedInstance(const DiversityProblem& dp,
                                         string strategy)
    : dp(dp), partition_strategy(strategy) {
  if (strategy == "all") {
    setup(strategy, dp.get_num_nodes());
  } else if (strategy == "none") {
    setup(strategy, 1);
  } else {
    cerr << "Invalid strategy! Or missing num_partitions!" << endl;
    return;
  }
}

PartitionedInstance::PartitionedInstance(const DiversityProblem& dp,
                                         string strategy, int num_partitions)
    : dp(dp), partition_strategy(strategy) {
  setup(strategy, num_partitions);
}

void PartitionedInstance::setup(string strategy, int num_partitions) {
  // Begin by gram-recovering locations and getting evecs and evals
  VectorXd evals;
  MatrixXd evecs;
  get_recovery_evals_evecs(evals, evecs);
  // Generate partition set
  generate_partition_set(strategy, num_partitions, evals);
  // Recover locations
  recover_locations(evals, evecs);
}

void PartitionedInstance::get_recovery_evals_evecs(VectorXd& evals,
                                                   MatrixXd& evecs) {
  // Generate Grammian
  int n = dp.get_num_nodes();
  MatrixXd gram(n, n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i; j < n; j++) {
      gram(i, j) = (dp.dist(i, 0) + dp.dist(j, 0) - dp.dist(i, j)) / 2.0;
      gram(j, i) = gram(i, j);
    }
  }
  // Pre-normalize
  if (pre_normalize_grammian) {
    MatrixXd row_sum = gram.rowwise().sum();
    double double_mean = gram.sum() / (double)n / (double)n;
    for (size_t i = 0; i < n; i++) {
      for (size_t j = i; j < n; j++) {
        gram(i, j) =
            gram(i, j) - (row_sum(i) + row_sum(j)) / (double)n + double_mean;
        gram(j, i) = gram(i, j);
      }
    }
  }
  if (gram.array().isNaN().any()) {
    cerr << "Grammian has NaNs! " << dp.get_name() << endl;
  }

  // Eigen decomposition
  Eigen::SelfAdjointEigenSolver<MatrixXd> solver(gram);
  if (solver.info() != Eigen::Success) {
    cerr << "Could not conduct eigenvalue decomposition! " << dp.get_name()
         << endl;
  }
  // Reverse order so largest eigenvalue is upfront
  evals = solver.eigenvalues().reverse();
  evecs = solver.eigenvectors().rowwise().reverse();

  // Get important eigenvalues and reorder
  int num_sig_evals = 0;
  for (int i = 0; i < evals.size(); ++i) {
    if (evals[i] < eigenvalue_tolerance) {
      break;
    } else {
      num_sig_evals++;
    }
  }
  if (num_sig_evals == 0) {
    std::cerr << "No coordinates!" << std::endl;
    return;
  }
  if (num_sig_evals < evals.size()) {
    evals = evals.head(num_sig_evals).eval();
    evecs = evecs.leftCols(num_sig_evals).eval();
  }
}
/*
Generate partition set based on strategy, and desired number of partitions.
Possible strategies:
 - "all" : Partitions ALL coordinates into their own partition.
 - "none" : Does not create any partitions, solves the original problem
 = "random" : Partitions of similar sizes but constains random coords
 - "stratified" : Attempts stratifed sampling.  Every partition explains a
                  similar amount, with intra-dissimilarly maximised
 - "greedy" : Just adds them in the given order into partitions of similar sizes
              Early partitions have very important coords.
 - "stepped" : Partitions of similar variance explained, but sizes may be very
               different.
 */
void PartitionedInstance::generate_partition_set(string strategy,
                                                 int num_partitions,
                                                 Eigen::VectorXd& evals) {
  /* Generate partition set, based on given strategy and desired number of
   * partitions */
  int num_coords = evals.size();
  if (strategy == "all" | num_coords <= num_partitions) {
    // All coordinates as their own partition
    partitions.resize(num_coords);
    for (size_t i = 0; i < num_coords; i++) {
      partitions[i].push_back(i);
    }

  } else if (strategy == "none" | num_partitions == 1) {
    // No partitions (original cutting plane)
    partitions.resize(1);
    for (size_t i = 0; i < num_coords; i++) {
      partitions[0].push_back(i);
    }

  } else if (strategy == "random") {
    // Random partitions
    vector<int> remaining_coordinates;
    for (size_t i = 0; i < num_coords; i++) remaining_coordinates.push_back(i);

    // Shuffle the vector to randomize it
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(remaining_coordinates.begin(), remaining_coordinates.end(), g);

    int c = 0;
    partitions.resize(num_partitions);
    for (size_t i = 0; i < num_coords % num_partitions; i++) {
      for (size_t j = 0; j < num_coords / num_partitions + 1; j++) {
        partitions[i].push_back(remaining_coordinates[c]);
        c++;
      }
    }
    for (size_t i = num_coords % num_partitions; i < num_partitions; i++) {
      for (size_t j = 0; j < num_coords / num_partitions; j++) {
        partitions[i].push_back(remaining_coordinates[c]);
        c++;
      }
    }

  } else if (strategy == "stratified") {
    // Attempts stratifed sampling.  Every partition explains a
    // similar amount, with intra-dissimilarly maximised
    partitions.resize(num_partitions);
    int c = 0;
    int p = 0;
    while (c < num_coords) {
      partitions[p].push_back(c);
      c++;
      p++;
      if (p == partitions.size()) p = 0;
    }
  } else if (strategy == "greedy") {
    // Just adds them in the given order into partitions of similar sizes
    // Early partitions have very important coords.
    partitions.resize(num_partitions);
    int c = 0;
    for (size_t i = 0; i < num_coords % num_partitions; i++) {
      for (size_t j = 0; j < num_coords / num_partitions + 1; j++) {
        partitions[i].push_back(c);
        c++;
      }
    }
    for (size_t i = num_coords % num_partitions; i < num_partitions; i++) {
      for (size_t j = 0; j < num_coords / num_partitions; j++) {
        partitions[i].push_back(c);
        c++;
      }
    }
  } else if (strategy == "stepped") {
    // Partitions of similar variance explained, but sizes may be very
    // different.
    int c = 0;
    int p = 0;
    double explained = 0.0;
    double sum_vals = evals.sum();
    while (c < num_coords) {
      vector<int> _par;
      while ((explained < 1 / (double)num_partitions * (p + 1)) &
             (c < num_coords)) {
        _par.push_back(c);
        explained += evals[c] / sum_vals;
        c++;
      }
      partitions.push_back(_par);
      p++;
    }
  } else {
    cerr << "Unknown strategy: " << strategy << endl;
  }

  // Check partitions
  vector<int> flatten_partitions;
  for (size_t i = 0; i < partitions.size(); i++) {
    for (int p : partitions[i]) flatten_partitions.push_back(p);
  }

  for (size_t i = 0; i < num_coords; i++) {
    bool valid = false;
    for (int c : flatten_partitions) {
      if (c == i) {
        valid = true;
        break;
      }
    }
    if (!valid) {
      cerr << "Invalid partition set! Missing coordinate! Strategy:"
           << partition_strategy << endl;
      break;
    }
  }
  if (num_coords != flatten_partitions.size()) {
    cerr << "Invalid partition set! Repated / extra coordinates! Strategy:"
         << partition_strategy << endl;
  }
}

void PartitionedInstance::recover_locations(VectorXd& evals, MatrixXd& evecs) {
  // Recover locations
  locations.resize(evecs.rows());
  for (int i = 0; i < evecs.rows(); i++) {
    locations[i].resize(partitions.size());
    for (int p = 0; p < partitions.size(); p++) {
      for (int j : partitions[p]) {
        locations[i][p].push_back(std::sqrt(std::abs(evals[j])) * evecs(i, j));
      }
    }
  }
}

double PartitionedInstance::dist(int par, int i, int j) const {
  return DiversityProblem::euclid_dist(locations[i][par], locations[j][par],
                                       true);
}

// Builds partitioned edm[partition][i][j]
void PartitionedInstance::build_partitioned_edm(
    vector<vector<vector<double>>>& edm) const {
  edm.resize(get_num_partitions());
  for (size_t p = 0; p < get_num_partitions(); p++) {
    edm[p].resize(get_num_nodes());
    for (size_t i = 0; i < get_num_nodes(); i++)
      edm[p][i].resize(get_num_nodes());
  }

  for (size_t p = 0; p < get_num_partitions(); p++) {
    for (size_t i = 0; i < get_num_nodes() - 1; i++) {
      for (size_t j = i + 1; j < get_num_nodes(); j++) {
        edm[p][i][j] = dist(p, i, j);
        edm[p][j][i] = edm[p][i][j];
      }
    }
  }
}

/*
Builds partitioned lower edm[partition][i][j].
Only defines the strict lower triangle!
```
 . . . .
 x . . .
 x x . .
 x x x .
```
therefore edm[p][i][j] is only defined when i > j.
If i < j, call edm[p][j][i].
*/
void PartitionedInstance::build_partitioned_lower_edm(
    vector<vector<vector<double>>>& edm) const {
  edm.resize(get_num_partitions());
  for (int p = 0; p < get_num_partitions(); p++) {
    edm[p].resize(get_num_nodes());
    for (int i = 0; i < get_num_nodes(); i++) {
      edm[p][i].resize(i);
      for (int j = 0; j < i; j++) {
        edm[p][i][j] = dist(p, i, j);
      }
    }
  }
}
