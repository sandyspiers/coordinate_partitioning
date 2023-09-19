#include "glover.hpp"

#include <chrono>
#include <iostream>

#include "diversity_problem.hpp"
#include "result.hpp"

using std::cerr;
using std::cout;
using std::endl;

GloverSolver::GloverSolver(const DiversityProblem& problem) : Solver(problem) {
  env = IloEnv();
  try {
    // Construct distance matrix
    problem.build_edm(edm);

    // Create model
    x = IloBoolVarArray(env, problem.get_num_coords());
    w = IloNumVarArray(env, problem.get_num_coords() - 1, 0.0, IloInfinity);

    // Choose m
    model.add(IloSum(x) == problem.get_cardinality());

    // Glover Reformulation
    for (int i = 0; i < problem.get_num_nodes() - 1; i++) {
      float Q_up = 0.0;
      for (int j = i + 1; j < problem.get_num_nodes(); j++) {
        Q_up += edm[i][j];
      }
      model.add(w[i] - x[i] * Q_up <= 0);

      IloExpr Q_x(env);
      for (int j = i + 1; j < problem.get_num_nodes(); j++) {
        Q_x += x[j] * edm[i][j];
      }
      model.add(w[i] - Q_x <= 0);
    }

    // Objective
    model.add(IloMaximize(env, IloSum(w)));

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

void GloverSolver::solve() {
  try {
    // Solver
    cplex = IloCplex(model);

    // Parameters
    cplex.setParam(IloCplex::Param::Threads, 1);
    cplex.setParam(IloCplex::Param::ClockType, 2);
    cplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap, 1e-9);
    cplex.setOut(env.getNullStream());
    cplex.setWarning(env.getNullStream());
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

Result GloverSolver::get_result(int timelimit, double solve_time) const {
  return Result(dp.get_name(), dp.get_num_nodes(), dp.get_num_coords(),
                dp.get_cardinality(), solver_name, "na", -1, -1, timelimit, -1,
                cplex.getObjValue(), cplex.getBestObjValue(),
                cplex.getMIPRelativeGap(), get_setup_time(), solve_time,
                get_setup_time() + solve_time);
}

Result GloverSolver::get_result(double timelimit, double obj_val,
                                double best_bound, double gap,
                                double solve_time) const {
  return Result(dp.get_name(), dp.get_num_nodes(), dp.get_num_coords(),
                dp.get_cardinality(), solver_name, "na", -1, -1, timelimit, -1,
                obj_val, best_bound, gap, get_setup_time(), solve_time,
                get_setup_time() + solve_time);
}