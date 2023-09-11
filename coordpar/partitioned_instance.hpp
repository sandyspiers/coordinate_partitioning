#ifndef PARTITION_H
#define PARTITION_H

#include <Eigen/Dense>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::string;
using std::vector;

class DiversityProblem;

class PartitionedInstance {
 private:
  // Parameters for locaiton gram-recovery
  const bool pre_normalize_grammian = true;
  const double eigenvalue_tolerance = 1e-6;

  // Set of partitions
  vector<vector<int>> partitions;
  // The recovered locations `locations[node][partition][coordinate]`
  vector<vector<vector<double>>> locations;

  // Setup procedure
  void setup(string strategy, int num_partitions);
  // Get recovery evecs & evals
  void get_recovery_evals_evecs(VectorXd &evals, MatrixXd &evec);
  // Generate partition set
  void generate_partition_set(string strategy, int num_partitions,
                              VectorXd &evals);
  // Recover locations by gram recovery
  void recover_locations(VectorXd &evals, MatrixXd &evec);

 public:
  // Original diversity problem instance
  const DiversityProblem &dp;
  const string partition_strategy;
  // Construct from diversity problem
  // Just for 'all' or 'none' where num_partitions is not important
  PartitionedInstance(const DiversityProblem &dp, string strategy);
  // For any strategy
  PartitionedInstance(const DiversityProblem &dp, string strategy,
                      int num_partitions);

  // Getters
  // Get location of node on given coordinate parititon
  const vector<double> &get_partitioned_location(int node,
                                                 int partition) const {
    return locations[node][partition];
  };
  // Get number partitions
  int get_num_partitions() const { return partitions.size(); };
  // Get num coords in a particular partition;
  int get_num_coords(int p) const { return partitions[p].size(); };
  // Get total number coordinates
  int get_num_coords() const {
    int sum = 0;
    for (size_t p = 0; p < get_num_partitions(); p++) sum += get_num_coords(p);
    return sum;
  };
  // Get number nodes
  int get_num_nodes() const { return locations.size(); };

  double dist(int par, int i, int j) const;
  void build_partitioned_edm(vector<vector<vector<double>>> &edm) const;
  void build_partitioned_lower_edm(vector<vector<vector<double>>> &edm) const;

  // Statics
};

#endif