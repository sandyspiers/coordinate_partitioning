import sys

import numpy as np
from cplex.callbacks import LazyConstraintCallback
from docplex.mp import model
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin

from .diversity_problem import DiversityProblem


class CutPlaneSolver:
    """
    Cut Plane solver for the EMSDP
    """

    def __init__(self, dp: DiversityProblem) -> None:
        """
        Constructs the cutting plane model for the EDMSP
        """
        self.dp = dp

        ## Construct integer model
        m = model.Model()

        m.theta = m.continuous_var(ub=self.dp.edm.sum(), name="theta")
        m.x = m.binary_var_list(self.dp.n, name="x")

        ## Choose p
        m.add_constraint(m.sum(m.x) == self.dp.p)

        ## Objective
        m.maximize(m.theta)

        ## Parameters
        m.parameters.mip.tolerances.mipgap = 0

        ## Save mdl
        self.m = m

        ## Add cuts
        self.tangents = self.m.register_callback(TangentPlanes)
        self.tangents.m = self.m
        self.tangents.dp = self.dp

    def solve(self, **kwargs):
        """
        Initiate solve procedure
        """
        self.m.solve(**kwargs)

        ## Get nodes and solution
        self.sol = np.array(self.m.solution.get_value_list(self.m.x))
        self.nodes = np.where(self.sol >= 1 - 1e-6)[0]

    def get_nodes(self):
        """
        Get list of nodes that solve problem
        """
        return self.nodes

    def get_sol(self):
        """
        Get problem solution as a numpy array
        """
        return self.sol


class TangentPlanes(ConstraintCallbackMixin, LazyConstraintCallback):
    """
    Outter approximation cuts for the EMSDP
    """

    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.nb_cuts = 0

    def __call__(self):
        try:
            self.nb_cuts += 1
            m = self.m
            dp: DiversityProblem = self.dp

            # fetch variable values to make solution of x
            y = np.array(self.make_solution_from_vars(m.x).get_value_list(m.x))

            fy = dp.f(y)
            dfy = dp.df(y)

            cut = m.theta <= fy - dfy.dot(y) + m.dot(m.x, dfy)
            coef, sence, rhs = self.linear_ct_to_cplex(cut)
            self.add(coef, sence, rhs)

        except:
            print(sys.exc_info()[0])
            raise
