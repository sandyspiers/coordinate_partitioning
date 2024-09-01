import numpy as np
import pandas as pd


def read_df(filename):
    df = pd.read_csv(
        filename,
        header=0,
        names=[
            "name",
            "n",
            "s",
            "p",
            "solver",
            "strategy",
            "recovered_coords",
            "partitions",
            "time_limit",
            "objval",
            "bound",
            "gap",
            "cuts",
            "setup_time",
            "solve_time",
            "total_time",
        ],
        dtype={
            "n": int,
            "s": int,
            "p": int,
            "recovered_coords": int,
            "partitions": int,
            "time_limit": int,
            "obj_val": float,
            "bound": float,
            "gap": float,
            "cuts": int,
            "setup_time": float,
            "solve_time": float,
            "total_time": float,
        },
        delimiter=", ",
        engine="python",
    )
    df.insert(8, "ratio", (df["partitions"] / df["n"]).round(2))
    df.loc[df["strategy"] == "all", "ratio"] = 1.0
    df.loc[df["solver"] != "coordpar", "ratio"] = 0
    df = df.sort_values(by=["solver", "strategy"])

    return df


def get_performance_profiles(df):
    """Returns a performance profile s of all unique solver setups within df"""
    # Get all unique sovers
    solvers = set(zip(df["solver"], df["strategy"], df["ratio"]))
    pp = {}
    for solver, strategy, ratio in solvers:
        times = [0] + list(
            df[
                (df["solver"] == solver)
                & (df["strategy"] == strategy)
                & (df["ratio"] == ratio)
                & (df["total_time"] <= df["time_limit"])
            ]["solve_time"].sort_values()
        )
        times.append(df["time_limit"].max())
        steps = np.arange(len(times))
        steps[-1] -= 1
        pp[solver, strategy, ratio] = (times, steps)
    return pp
