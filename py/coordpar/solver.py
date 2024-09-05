import math
import sys

import numpy as np
from cplex.callbacks import LazyConstraintCallback
from docplex.mp import model
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from scipy.spatial.distance import cdist

from . import euclid, partition
from .diversity_problem import DiversityProblem


class CoordinatePartitionSolver:
    """
    Coordinate partition solver.
    """

    def __init__(
        self, dp: DiversityProblem, strategy, ratio=None, low_memory_mode=False
    ) -> None:
        """
        Base construcor for the coordinate partition solver.

        Parameters:
         - dp (DiversityProblem) : Problem instance
         - strategy (string) : Partition strategy.  Options include...
         - partition_ratio (float) : Ratio of num partitions to num coords.
         - low_memory_mode (bool) : Whether or not to use low memory cut generation
        """

        # Public variables
        self.dp = dp
        self.strategy = strategy
        self.partition_ratio = ratio
        self.low_memory_mode = low_memory_mode
        self.sol = None
        self.nodes = None

        # Recover locations
        self.locations, evals = euclid.gram_recovery(self.dp.edm, return_evals=True)

        # Determine number of partitions based on ratio
        self.num_coords = self.locations.shape[1]
        self.num_partitions = min(self.num_coords, math.ceil(dp.n * ratio))

        # Get partitions
        self.partitions = partition.from_strategy(
            strategy, self.num_partitions, self.num_coords, evals
        )

        # Update number of partitions (can change depending on partition strategy)
        self.num_partitions = len(self.partitions)

        # Generate edms
        if self.low_memory_mode:
            self.edms = euclid.build_edm(self.locations)
        else:
            self.edms = euclid.build_dissaggregated_edm(self.locations)
            self.edms = euclid.partition_dissaggregated_edm(self.edms, self.partitions)

        # Construct model
        self.m = model.Model()
        self.m.x = self.m.binary_var_list(self.dp.n, name="x")
        self.m.theta = self.m.continuous_var_list(
            self.num_partitions, ub=self.edms.sum(), name="theta"
        )

        # Choose p
        self.m.add_constraint(self.m.sum(self.m.x) == self.dp.p)

        # Objective
        self.m.maximize(self.m.sum(self.m.theta))

        # Parameters
        self.m.parameters.threads = 1
        self.m.parameters.mip.tolerances.mipgap = 0

        ## Add cuts
        self.tangents = self.m.register_callback(CoordinatePartitionTangents)
        self.tangents.solver = self

    def solve(self, **kwargs):
        """
        Solve by branch and cut
        """
        ## Solve
        sol = self.m.solve(**kwargs)

        ## Get nodes and solution
        if sol is None:
            return
        self.sol = np.array(sol.get_value_list(self.m.x))
        self.nodes = np.where(self.sol >= 1 - 1e-6)[0]

    def get_nodes(self):
        return self.nodes

    def get_sol(self):
        return self.sol


class CoordinatePartitionTangents(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.nb_cuts = 0
        self.solver: CoordinatePartitionSolver = None

    def __call__(self):
        try:
            # Shorthanding...
            self.nb_cuts += 1
            m = self.solver.m
            p = self.solver.dp.p

            # Get solution...
            y = np.array(self.make_solution_from_vars(m.x).get_value_list(m.x))

            if self.solver.low_memory_mode:
                # Add a cut for each partition
                wy = np.where(y == 1)[0]
                for c in range(len(self.solver.partitions)):
                    # Calculate fy
                    par = self.solver.partitions[c]
                    locp = self.solver.locations[:, par]

                    mags = np.square(locp).sum(axis=1)
                    gram = locp.dot(locp[wy, :].T)

                    mag_y_sum = mags[wy].sum()

                    fy = p * mag_y_sum - gram[wy].sum()
                    dy = p * mags + mag_y_sum - 2 * gram.sum(axis=1)
                    self.add(
                        *self.linear_ct_to_cplex(
                            m.theta[c] <= -fy + m.scal_prod(m.x, dy)
                        )
                    )

            else:
                # Determine f and dy matricies, this will include values over all partitions
                dy = y.dot(self.solver.edms)
                fy = y.dot(dy) / 2

                for c in range(len(fy)):
                    self.add(
                        *self.linear_ct_to_cplex(
                            m.theta[c] <= -fy[c] + m.scal_prod(m.x, dy[:, c])
                        )
                    )

        except:
            print(sys.exc_info()[0])
            raise
