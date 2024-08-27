#ifndef DIVERSITY_PROBLEM_H
#define DIVERSITY_PROBLEM_H

#include <string>
#include <vector>

using std::string;
using std::vector;
/*
Holds a single instance of the Euclidean Max-Sum Diversity Problem.
This includes:
 - Number nodes
 - Desired cardinality
 - Number coordinates
 - Original locations

An instance can be constructed by either reading a file, or generating a random
instance.
DOES NOT contain the EDM!
 */
class DiversityProblem {
 private:
  int num_nodes = 0;    // Total Number Nodes
  int cardinality = 0;  // Desired cardinality
  int num_coords = 0;   // Number coordinates
  string name = "dp";
  int seed;

  vector<vector<double>> locations;  // Locations
 public:
  // Empty constructor
  DiversityProblem(){};
  // Random constructors with types
  DiversityProblem(string type, int num_nodes, int cardinality, int num_coords,
                   int seed = -1);
  void random_box(int num_nodes, int cardinality, int num_coords,
                  int axis_limit = 100);
  void random_circle(int num_nodes, int cardinality, int num_coords,
                     int diameter = 100);
  // Calls the 'from_file' method, but must provide a cardinality after calling!
  DiversityProblem(string file_name);
  void from_file(string file_name);

  // sets the cardinality (For re-use and from_file constructors)
  void set_cardinality(int cardinality) { this->cardinality = cardinality; };

  // Getters
  int get_num_nodes() const { return num_nodes; }
  int get_cardinality() const { return cardinality; }
  int get_num_coords() const { return num_coords; }
  const string &get_name() const { return name; }

  const vector<double> &get_location(int i) const { return locations[i]; }
  const vector<vector<double>> &get_locations() const { return locations; }

  double dist(int i, int j, bool squared = false) const;
  void build_edm(vector<vector<double>> &edm, bool squared = false) const;
  void build_lower_edm(vector<vector<double>> &edm, bool squared = false) const;

  // Static euclidean distance helper function
  static double euclid_dist(const vector<double> &a, const vector<double> &b,
                            bool squared);
};

#endif
