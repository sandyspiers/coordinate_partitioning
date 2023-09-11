#include "result.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

#include "diversity_problem.hpp"
#include "solver.hpp"

// All explicit
Result::Result(const string& problem_name, int num_nodes, int num_coords,
               int cardinality, const string& solver_name,
               const string& strategy, int num_recovered_coords,
               int num_partitions, int timelimit, int num_cuts, double obj_val,
               double best_bound, double gap, double setup_time,
               double solve_time, double total_time)
    : problem_name(problem_name),
      num_nodes(num_nodes),
      num_coords(num_coords),
      cardinality(cardinality),
      solver_name(solver_name),
      strategy(strategy),
      num_recovered_coords(num_recovered_coords),
      num_partitions(num_partitions),
      timelimit(timelimit),
      num_cuts(num_cuts),
      obj_val(obj_val),
      best_bound(best_bound),
      gap(gap),
      setup_time(setup_time),
      solve_time(solve_time),
      total_time(total_time) {}

string Result::get_string() const {
  std::stringstream lineStream;
  // Write all variables, separated by commas
  lineStream << std::setprecision(15) << problem_name << ", " << num_nodes
             << ", " << num_coords << ", " << cardinality << ", " << solver_name
             << ", " << strategy << ", " << num_recovered_coords << ", "
             << num_partitions << ", " << timelimit << ", " << obj_val << ", "
             << best_bound << ", " << gap << ", " << num_cuts << ", "
             << setup_time << ", " << solve_time << ", " << total_time;
  return lineStream.str();
}

void Result::write(string filename) const {
  std::ofstream outfile;
  // Open file in append mode
  outfile.open(filename, std::ios_base::app);
  if (!outfile.is_open()) {
    // Handle the error (e.g., throw an exception or print an error message)
    std::cerr << "Unable to open the file: " << filename << std::endl;
    return;
  }
  outfile << get_string() << std::endl;
  // Close the file
  outfile.close();
};

void IntermediateResultsI::main() {
  if (getCplexTime() - start_time > timelimits.front() &&
      timelimits.size() > 0) {
    // We have hit the next time limit
    Result res = solver->get_result(timelimits.front(), getIncumbentObjValue(),
                                    getBestObjValue(), getMIPRelativeGap(),
                                    getCplexTime() - start_time);
    results.push_back(res);
    timelimits.erase(timelimits.begin());
    std::cout << "result... " << res.get_string() << std::endl;
    if (timelimits.size() == 0) {
      abort();
    }
  }
  return;
};

IloCplex::Callback IntermediateResults(IloEnv env, IloNum& start_time,
                                       vector<int>& timelimits,
                                       vector<Result>& results,
                                       Solver* solver) {
  return (IloCplex::Callback(new (env) IntermediateResultsI(
      env, start_time, timelimits, results, solver)));
}
