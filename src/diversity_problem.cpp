#include "diversity_problem.hpp"

#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>

using std::cerr;
using std::cout;
using std::endl;

// Calculate the squared distance between two vectors
double DiversityProblem::euclid_dist(const vector<double> &a,
                                     const vector<double> &b, bool squared) {
  if (a.size() != b.size()) {
    std::cerr << "Error: Vectors must be of the same size for distance "
                 "calculation.\n";
    return -1;
  }
  double _dist = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double diff = a[i] - b[i];
    _dist += diff * diff;
  }
  if (squared) return _dist;
  return std::sqrt(_dist);
}

// Get the distance between locations i and j
double DiversityProblem::dist(int i, int j, bool squared) const {
  return euclid_dist(locations[i], locations[j], squared);
}

/*
Builds the euclidean distance matrix based on locations, and saves it to the
user provided `edm`.
 */
void DiversityProblem::build_edm(vector<vector<double>> &edm,
                                 bool squared) const {
  edm.resize(locations.size(), vector<double>(locations.size(), 0.0));
  for (int i = 0; i < locations.size(); ++i) {
    for (int j = i + 1; j < locations.size(); ++j) {
      double _dist = dist(i, j, squared);
      edm[i][j] = _dist;
      edm[j][i] = _dist;  // Distance matrix is symmetric
    }
  }
}

/*
Builds the euclidean distance matrix based on locations, and saves it to the
user provided `edm`. Only defines the strict lower triangle!
```
 . . . .
 x . . .
 x x . .
 x x x .
```
therefore edm[i][j] is only defined when i > j.
If i < j, call edm[j][i].
 */
void DiversityProblem::build_lower_edm(vector<vector<double>> &edm,
                                       bool squared) const {
  edm.resize(locations.size());
  for (int i = 0; i < locations.size(); ++i) {
    edm[i].resize(i);
    for (int j = 0; j < i; ++j) {
      edm[i][j] = dist(i, j, squared);
    }
  }
}

/*
Random constructor.
Checks the type, then calls the constructor routine of the same name.
Options include:
 - random
 - circle
 */
DiversityProblem::DiversityProblem(string type, int num_nodes, int cardinality,
                                   int num_coords, int seed) {
  name = "dp_" + type;
  if (seed < 0) {
    this->seed = static_cast<int>(std::time(nullptr));
  } else {
    this->seed = seed;
    name += "_" + std::to_string(seed);
  }
  if (type == "random" | type == "box") {
    random_box(num_nodes, cardinality, num_coords);
  } else if (type == "circle") {
    random_circle(num_nodes, cardinality, num_coords);
  } else {
    cerr << "Unknown type: " << type << endl;
  }
}

/*
Generates `locations` random, where every location has same number
coordinates. Every coordinate is uniformly randomly generated in the range
[0,axis_limit].
*/
void DiversityProblem::random_box(int num_nodes, int cardinality,
                                  int num_coords, int axis_limit) {
  std::default_random_engine gen(seed);
  std::uniform_real_distribution<double> U(0.0,
                                           static_cast<double>(axis_limit));
  this->num_nodes = num_nodes;
  this->cardinality = cardinality;
  this->num_coords = num_coords;
  locations.resize(num_nodes);

  for (int i = 0; i < num_nodes; ++i) {
    locations[i].resize(num_coords);
    for (int j = 0; j < num_coords; ++j) {
      locations[i][j] = U(gen);
    }
  }
}

/*
Generates `locations` random, where every location has same number
coordinates. Every location is randomly generated somewhere along the edge of
a circle of a given diameter.
*/
void DiversityProblem::random_circle(int num_nodes, int cardinality,
                                     int num_coords, int diameter) {
  std::default_random_engine gen(seed);
  std::normal_distribution<double> G(0.0, 1.0);
  this->num_nodes = num_nodes;
  this->cardinality = cardinality;
  this->num_coords = num_coords;
  locations.resize(num_nodes);

  for (int i = 0; i < num_nodes; ++i) {
    locations[i].resize(num_coords);
    double norm = 0.0;
    for (int j = 0; j < num_coords; ++j) {
      // randomly generate coordinate from gaussian distribution
      locations[i][j] = G(gen);
      norm += locations[i][j] * locations[i][j];
    }
    norm = std::sqrt(norm);
    for (int j = 0; j < num_coords; j++) {
      locations[i][j] = locations[i][j] / norm * diameter / 2.0;
    }
  }
}

/*
Reads the provided file, which should be of the format
```
num_nodes
x11 x12 x12 ...
x21 x22 x23 ...
```
where each row contains a new location, and each column are the coordinates of
the locations. This information is then read into `locations`.
*/
DiversityProblem::DiversityProblem(string file_name) { from_file(file_name); }

// Reads instance from file
void DiversityProblem::from_file(string file_name) {
  std::ifstream file(file_name);
  if (!file.is_open()) {
    cerr << "Could not open file: " << file_name << endl;
    return;
  }

  file >> num_nodes;
  locations.resize(num_nodes);

  for (int i = 0; i < num_nodes; ++i) {
    locations[i].resize(num_coords);
    for (int j = 0; j < num_coords; ++j) {
      file >> locations[i][j];
    }
  }
  file.close();
}
