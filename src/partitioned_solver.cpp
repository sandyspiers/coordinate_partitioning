#include "partitioned_solver.hpp"

#include "diversity_problem.hpp"

using std::cerr;
using std::cout;
using std::endl;

// An optimised way of generating tangents of quadratic functions.
void create_tangent(IloExpr& tangent, const IloBoolVarArray& x,
                    const IloNumArray& y, const int par,
                    const function<const double(int, int, int)>& get_dist) {
  // cut: -f(y) + <df(y),x>
  int n = y.getSize();
  for (int i = 0; i < n; i++) {
    if (y[i] >= 1 - 1e-12) {
      for (int j = 0; j < i; j++) {
        tangent += get_dist(par, i, j) * x[j];
      }
      for (int j = i + 1; j < n; j++) {
        tangent += get_dist(par, i, j) * x[j];
        if (y[j] >= 1 - 1e-12) tangent -= get_dist(par, i, j);
      }
    }
  }
}

ILOLAZYCONSTRAINTCALLBACK4(PartitionedTangentPlanes, IloBoolVarArray, x,
                           IloNumVarArray, theta,
                           const function<const double(int, int, int)>&,
                           get_dist, int&, num_cuts) {
  // Get environment
  IloEnv env = getEnv();
  // Get x solution
  IloNumArray y(env, x.getSize());
  getValues(y, x);
  // Generate and add cut for each partition
  for (size_t p = 0; p < theta.getSize(); p++) {
    IloExpr cut(env);
    create_tangent(cut, x, y, p, get_dist);
    num_cuts += 1;
    add(theta[p] <= cut).end();
    cut.end();
  }
  y.end();
}

CoordinatePartitionSolver::CoordinatePartitionSolver(const DiversityProblem& dp,
                                                     string strategy,
                                                     bool low_memory_cuts)
    : Solver(dp), pi(dp, strategy), low_memory_cuts(low_memory_cuts) {
  setup_model();
};
CoordinatePartitionSolver::CoordinatePartitionSolver(const DiversityProblem& dp,
                                                     string strategy,
                                                     int num_partitions,
                                                     bool low_memory_cuts)
    : Solver(dp),
      pi(dp, strategy, num_partitions),
      low_memory_cuts(low_memory_cuts) {
  setup_model();
};
CoordinatePartitionSolver::CoordinatePartitionSolver(const DiversityProblem& dp,
                                                     string strategy,
                                                     double partition_ratio,
                                                     bool low_memory_cuts)
    : Solver(dp),
      pi(dp, strategy, dp.get_num_nodes() * partition_ratio),
      low_memory_cuts(low_memory_cuts) {
  setup_model();
};

void CoordinatePartitionSolver::setup_model() {
  env = IloEnv();
  try {
    num_cuts = 0;
    model = IloModel(env);
    x = IloBoolVarArray(env, pi.get_num_nodes());
    theta = IloNumVarArray(env, pi.get_num_partitions(), 0.0, IloInfinity);

    // Choose m
    model.add(IloSum(x) == pi.dp.get_cardinality());

    // Objective
    model.add(IloMaximize(env, IloSum(theta)));

    // If need euclidean distance matrix set it up now
    if (low_memory_cuts) {
      get_dist = [this](int p, int i, int j) -> double {
        return this->pi.dist(p, i, j);
      };
    } else {
      pi.build_partitioned_lower_edm(edm);
      get_dist = [this](int p, int i, int j) -> double {
        if (i > j) return this->edm[p][i][j];
        if (i < j) return this->edm[p][j][i];
        return 0.0;
      };
    }

    // Add in the first cut (avoids upper bounding theta)
    IloNumArray y(env, dp.get_num_nodes());
    for (size_t i = 0; i < dp.get_cardinality(); i++) {
      y[i] = 1.0;
    }
    for (size_t p = 0; p < pi.get_num_partitions(); p++) {
      IloExpr cut(env);
      create_tangent(cut, x, y, p, get_dist);
      model.add(theta[p] <= cut);
    }
    y.end();

  } catch (IloException& ex) {
    cerr << "Error: " << ex << endl;
  } catch (...) {
    cerr << "Error" << endl;
  }
  auto end_setup_time = std::chrono::high_resolution_clock::now();
  setup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                   end_setup_time - start_time)
                   .count() /
               1000.0;
}

void CoordinatePartitionSolver::solve() {
  try {
    // Solver
    cplex = IloCplex(model);

    // Tangents
    cplex.use(PartitionedTangentPlanes(env, x, theta, get_dist, num_cuts));

    // Parameters
    cplex.setParam(IloCplex::Param::Threads, 16);
    cplex.setParam(IloCplex::Param::ClockType, 2);
    cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 1e-9);
    cplex.setParam(IloCplex::Param::MIP::Strategy::File, 1);
    // cplex.setOut(env.getNullStream());
    // cplex.setWarning(env.getNullStream());
    // cplex.setError(env.getNullStream());

    // Solve with timers
    IloNum solve_time;
    if (timelimits.size() > 0) {
      cplex.use(
          IntermediateResults(env, solve_time, timelimits, results, *this));
    }
    solve_time = cplex.getCplexTime();
    cplex.solve();
    solve_time = cplex.getCplexTime() - solve_time;

    // Report for the remaining time limits
    while (timelimits.size() > 0) {
      results.push_back(get_result(timelimits.front(), solve_time));
      timelimits.erase(timelimits.begin());
    }

  } catch (IloException& ex) {
    cerr << "Error: " << ex << endl;
  } catch (...) {
    cerr << "Error" << endl;
  }
}

Result CoordinatePartitionSolver::get_result(int timelimit,
                                             double solve_time) const {
  return Result(dp.get_name(), dp.get_num_nodes(), dp.get_num_coords(),
                dp.get_cardinality(), solver_name, pi.partition_strategy,
                pi.get_num_coords(), pi.get_num_partitions(), timelimit,
                num_cuts, cplex.getObjValue(), cplex.getBestObjValue(),
                cplex.getMIPRelativeGap(), get_setup_time(), solve_time,
                get_setup_time() + solve_time);
}

Result CoordinatePartitionSolver::get_result(double timelimit, double obj_val,
                                             double best_bound, double gap,
                                             double solve_time) const {
  return Result(dp.get_name(), dp.get_num_nodes(), dp.get_num_coords(),
                dp.get_cardinality(), solver_name, pi.partition_strategy,
                pi.get_num_coords(), pi.get_num_partitions(), timelimit,
                num_cuts, obj_val, best_bound, gap, get_setup_time(),
                solve_time, get_setup_time() + solve_time);
}
