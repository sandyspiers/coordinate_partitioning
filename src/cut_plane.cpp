#include "cut_plane.hpp"

#include <chrono>
#include <iostream>

#include "diversity_problem.hpp"
#include "result.hpp"

using std::cerr;
using std::cout;
using std::endl;

// An optimised way of generating tangents of quadratic functions.
void create_tangent(IloExpr& tangent, const IloBoolVarArray& x,
                    const IloNumArray& y,
                    const function<const double(int, int)>& get_dist) {
  // cut: -f(y) + <df(y),x>
  for (int i = 0; i < y.getSize(); i++) {
    if (y[i] >= 1 - 1e-12) {
      for (int j = 0; j < i; j++) {
        tangent += get_dist(i, j) * x[j];
      }
      for (int j = i + 1; j < y.getSize(); j++) {
        tangent += get_dist(i, j) * x[j];
        if (y[j] >= 1 - 1e-12) tangent -= get_dist(i, j);
      }
    }
  }
}

ILOLAZYCONSTRAINTCALLBACK4(TangentPlanes, IloBoolVarArray, x, IloNumVar, theta,
                           const function<const double(int, int)>&, get_dist,
                           int&, num_cuts) {
  // Get environment
  IloEnv env = getEnv();
  // Get x solution
  IloNumArray y(env, x.getSize());
  getValues(y, x);
  // Generate and add cut
  IloExpr cut(env);
  create_tangent(cut, x, y, get_dist);
  num_cuts++;
  add(theta <= cut).end();
  cut.end();
  y.end();
}

CutPlaneSolver::CutPlaneSolver(const DiversityProblem& problem,
                               const bool low_memory_cuts)
    : Solver(problem), low_memory_cuts(low_memory_cuts) {
  env = IloEnv();
  try {
    // Create model
    model = IloModel(env);
    x = IloBoolVarArray(env, dp.get_num_nodes());
    theta = IloNumVar(env, "theta");

    // Choose m
    model.add(IloSum(x) == dp.get_cardinality());

    // Objective
    model.add(IloMaximize(env, theta));

    // If need euclidean distance matrix set it up now
    if (low_memory_cuts) {
      get_dist = [this](int i, int j) -> double { return this->dp.dist(i, j); };
    } else {
      dp.build_lower_edm(edm);
      get_dist = [this](int i, int j) -> double {
        if (i > j) return this->edm[i][j];
        if (i < j) return this->edm[j][i];
        return 0.0;
      };
    }

    // Add in the first cut (avoids upper bounding theta)
    IloNumArray y(env, dp.get_num_nodes());
    for (size_t i = 0; i < dp.get_cardinality(); i++) {
      y[i] = 1.0;
    }
    IloExpr cut(env);
    create_tangent(cut, x, y, get_dist);
    model.add(theta <= cut);
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

void CutPlaneSolver::solve() {
  try {
    // Solver
    cplex = IloCplex(model);

    // Tangents
    cplex.use(TangentPlanes(env, x, theta, get_dist, num_cuts));

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

Result CutPlaneSolver::get_result(int timelimit, double solve_time) const {
  return Result(dp.get_name(), dp.get_num_nodes(), dp.get_num_coords(),
                dp.get_cardinality(), solver_name, "na", -1, -1, timelimit,
                num_cuts, cplex.getObjValue(), cplex.getBestObjValue(),
                cplex.getMIPRelativeGap(), get_setup_time(), solve_time,
                get_setup_time() + solve_time);
}

Result CutPlaneSolver::get_result(double timelimit, double obj_val,
                                  double best_bound, double gap,
                                  double solve_time) const {
  return Result(dp.get_name(), dp.get_num_nodes(), dp.get_num_coords(),
                dp.get_cardinality(), solver_name, "na", -1, -1, timelimit,
                num_cuts, obj_val, best_bound, gap, get_setup_time(),
                solve_time, get_setup_time() + solve_time);
}
