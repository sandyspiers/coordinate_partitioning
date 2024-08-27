import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from .cut_plane_solver import CutPlaneSolver
from .diversity_problem import DiversityProblem


def get_exclusion_zones(
    dp: DiversityProblem, y, lb=None, squared=False, precision=1000, space_increase=0
):
    """
    Takes the solution y, and calculates the exclusion zones

    Returns X,Y,Z where X,Y are meshgrid and Z is list of exclusion zones values.
    """

    fy = dp.f(y)
    dfy = dp.df(y)
    if lb is None:
        lb = fy

    # Get distances
    grid = np.linspace(
        dp.locations.min() - space_increase,
        dp.locations.max() + space_increase,
        precision,
    )
    X, Y = np.meshgrid(grid, grid)

    # Calculate distances using cdist
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    distances = cdist(grid_points, dp.locations[y == 1])
    if squared:
        distances = np.square(distances)

    # Sum distances for each point
    sum_distances = np.sum(distances, axis=1)

    # Reshape the result back to the grid shape
    Z_dist = sum_distances.reshape(X.shape)

    Z = []
    for rp in np.where(y == 1)[0]:
        Z.append(Z_dist <= (lb - fy) / dp.p + dfy[rp])
    Z = np.array(Z)

    return X, Y, Z


def get_proportion_solutions_removed(dp: DiversityProblem, y, lb=None):
    """
    Determines the proportion of removed solutions by nested sum method
    """

    # Setup...
    fy = dp.f(y)
    dfy = dp.df(y)
    if lb is None:
        lb = fy
    ordered = np.argsort(dfy[y == 1])
    h = [np.sum(dfy <= (lb - fy) / dp.p + dfy[y == 1][j]) for j in ordered]

    # Inner sum function
    def inner_sum(k: int, sum_prev_i: int):
        # Start the sum...
        _sum = 0
        for i in range(
            max(k - sum_prev_i, 0), min(h[k - 1] - h[k - 1 - 1], dp.p - sum_prev_i) + 1
        ):
            s1 = math.comb(h[k - 1] - h[k - 1 - 1], i)
            if sum_prev_i + i == dp.p:
                s2 = 1
            else:
                s2 = inner_sum(k + 1, sum_prev_i + i)
            _sum += s1 * s2
        return _sum

    # Go through inner sums
    r_suff_loop_sum = 0
    for i in range(1, min(h[0], dp.p) + 1):
        r_suff_loop_sum += math.comb(h[0], i) * inner_sum(2, i)

    return r_suff_loop_sum / math.comb(dp.n, dp.p)


def get_proportion_solutions_removed_combinatorial(dp: DiversityProblem, y, lb=None):
    """
    Gets the proportion of solutions removed by y, using brute force.
    """
    sols = []
    for combo in itertools.combinations(range(dp.n), dp.p):
        vector = np.zeros(dp.n, dtype=int)
        vector[list(combo)] = 1
        sols.append(vector)
    sols = np.array(sols)

    fy = dp.f(y)
    dfy = dp.df(y)
    if lb is None:
        lb = fy

    cut_vals = fy - dfy.dot(y) + sols.dot(dfy)

    return np.sum(cut_vals <= lb) / math.comb(dp.n, dp.p)


def get_heur_sol(dp: DiversityProblem, timelimit=2):
    # Solve
    ct = CutPlaneSolver(dp)
    ct.m.parameters.threads = 1
    ct.m.parameters.timelimit = timelimit
    ct.solve()
    return ct.sol


def get_rand_sol(dp: DiversityProblem):
    """Get a random carindality solution"""
    y = np.zeros(dp.n)
    y[np.random.choice(dp.n, dp.p, replace=False)] = 1
    return y


def plot_zones_onto_axes(axes, y, X, Y, Z, dp: DiversityProblem):
    """Plot exclusion zones onto given axis"""
    # Colour map
    CM = plt.get_cmap("Reds")(np.linspace(0.4, 1.0, dp.p))
    # Ordered df points
    ordered = np.argsort(dp.df(y)[y == 1])

    # Plot other locations
    axes.plot(
        dp.locations[y == 0, 0],
        dp.locations[y == 0, 1],
        ls="",
        marker="o",
        alpha=0.5,
    )

    # Plot each location and its contour
    k = 0
    for i in ordered:
        axes.contour(X, Y, Z[i], linestyles="dashed", colors=[CM[k]])
        axes.scatter(
            dp.locations[y == 1, 0][i], dp.locations[y == 1, 1][i], color=CM[k]
        )
        k += 1

    return axes
