import numpy as np


def split_sizes(length, num_splits):
    """
    Returns the sizes of the $m$ partitions of a list of size $n$.
    Sizes are similar, only differing by 1.
    Larger sizes appear first.
    """
    assert length >= num_splits
    sizes = np.zeros(num_splits, dtype=int)
    sizes[: length % num_splits] = length // num_splits + 1
    sizes[length % num_splits :] = length // num_splits
    return sizes


def similar_sized_similar_variance(**kwargs):
    """
    Returns a list of indices that creates a partitions of the coordinates of locations.
    Attempts to create partitions of similar sizes and similar variance explained.
    Assumes that coordinates of locations are ordered from most important to least.

    Parameters:
     - num_partitions
     - num_coords
    """
    # If no kwargs, returns the names of this function
    if not kwargs:
        # Return names
        return ["similar_sized_similar_variance", "sssv"]
    # Otherwise, attempt to get kwargs
    try:
        num_partitions = kwargs["num_partitions"]
        num_coords = kwargs["num_coords"]
    except TypeError:
        raise TypeError("Need num_partitions and num_coords")

    ## Get sizes of splits, from SMALLEST to largest
    splt_sz = split_sizes(num_coords, num_partitions)[::-1]

    ## Create partitions obj
    partitions = [[] for k in range(num_partitions)]

    ## Fill in the coordinates by dropping them into different bins.
    c = 0
    while c < num_coords:  # for every coordinate
        for k in range(num_partitions):
            if len(partitions[k]) < splt_sz[k] and c < num_coords:
                # drop into a partion not filled
                # put coordinate c into this partition
                partitions[k].append(c)
                c += 1

    ## Return partitions
    return [np.array(p) for p in partitions]


def similar_sized_greedy_variances(**kwargs):
    """
    Returns a list of indices that creates a partitions of the coordinates of locations.
    Attempts to create partitions of similar sizes, but greedy based on variance.
    Should have big range in variance explained, from most to least.
    Assumes that coordinates of locations are ordered from most important to least.

    Parameters:
     - num_partitions
     - num_coords
    """
    # If no kwargs, returns the names of this function
    if not kwargs:
        # Return names
        return ["similar_sized_greedy_variances", "ssgv"]
    # Otherwise, attempt to get kwargs
    try:
        num_partitions = kwargs["num_partitions"]
        num_coords = kwargs["num_coords"]
    except TypeError:
        raise TypeError("Need num_partitions and num_coords")

    splt_sz = split_sizes(num_coords, num_partitions)
    partitions = []
    c = 0
    for k in range(num_partitions):
        partitions.append(np.arange(c, c + splt_sz[k]))
        c += splt_sz[k]
    return partitions


def greedy_coordinates_similar_variances(**kwargs):
    """
    Returns a list of indices that creates a partitions of the coordinates of locations.
    Attempts create partitions that explain similar number of variance,
    but greedy number of coordinates.
    First partitions have few coordinates, last partition has the most.
    Requires the eigenvalues.
    Assumes that coordinates of locations are ordered from most important to least.

    Parameters:
     - num_partitions
     - num_coords
     - vals
    """
    # If no kwargs, returns the names of this function
    if not kwargs:
        # Return names
        return ["greedy_coordinates_similar_variances", "gcsv"]
    # Otherwise, attempt to get kwargs
    try:
        num_partitions = kwargs["num_partitions"]
        num_coords = kwargs["num_coords"]
        vals = kwargs["vals"]
    except TypeError:
        raise TypeError("Need num_partitions, num_coords and eigenvals")

    partitions = [[] for k in range(num_partitions)]
    c = 0
    explained = 0
    for k in range(num_partitions):
        while explained < 1 / num_partitions * (k + 1) and c < num_coords:
            partitions[k].append(c)
            explained += vals[c] / vals.sum()
            c += 1
    return [p for p in partitions if len(p) > 0]


def similar_sized_random(**kwargs):
    """
    Returns a list of indices that creates a partitions of the coordinates of locations.
    Attempts to have a similar number of coordinates, but chooses the coordinates at random.
    Good as a control test.
    Assumes that coordinates of locations are ordered from most important to least.

    Parameters:
     - num_partitions
     - num_coords
    """
    # If no kwargs, returns the names of this function
    if not kwargs:
        # Return names
        return ["similar_sized_random", "ssr", "rand", "random"]
    # Otherwise, attempt to get kwargs
    try:
        num_partitions = kwargs["num_partitions"]
        num_coords = kwargs["num_coords"]
    except TypeError:
        raise TypeError("Need num_partitions and num_coords")

    splt_sz = split_sizes(num_coords, num_partitions)
    coords_left = np.arange(num_coords)
    partitions = []
    for k in range(num_partitions):
        # Choose some random indices
        random_coords_indices = np.random.choice(
            coords_left.shape[0], splt_sz[k], replace=False
        )
        # Get the chosen coords
        random_coords = coords_left[random_coords_indices]
        # Remove from set
        coords_left = np.delete(coords_left, random_coords_indices)
        # Add to partitions
        partitions.append(random_coords)
    return partitions


def all_partition(**kwargs):
    """
    Returns a list of indices that creates a partitions of the coordinates of locations.
    Every partion is just 1 coordinate.
    Assumes that coordinates of locations are ordered from most important to least.

    Parameters:
     - num_coords
    """
    # If no kwargs, returns the names of this function
    if not kwargs:
        # Return names
        return ["all_partition", "all"]
    # Otherwise, attempt to get kwargs
    try:
        num_coords = kwargs["num_coords"]
    except TypeError:
        raise TypeError("Need num_coords")

    return [[c] for c in range(num_coords)]


def no_partition(**kwargs):
    """
    Puts all coordinates into one

    Parameters:
     - num_coords
    """
    # If no kwargs, returns the names of this function
    if not kwargs:
        # Return names
        return ["no_partition", "no", "none"]
    # Otherwise, attempt to get kwargs
    try:
        num_coords = kwargs["num_coords"]
    except TypeError:
        raise TypeError("Need num_coords")

    return [np.arange(num_coords)]


def from_strategy(
    strategy: str, num_partitions: int, num_coords: int, vals
) -> list[list]:
    """
    A general procedure to return the partitions based on the selected strategy.
    Checks that the strategy is valid, the num partitions are not more than the num coords.
    """
    strategy = strategy.lower()
    if strategy not in STRATEGY_DICTIONARY.keys():
        raise Exception(
            f"Please choose a valid strategy from {STRATEGY_DICTIONARY.keys()}"
        )
    if num_partitions > num_coords:
        raise Exception(
            f"Too many partitions! {num_partitions} partitions but on {num_coords} number coordinates!"
        )
    if num_partitions < 0:
        raise Exception(f"Number of partitions must be nonnegative")
    return STRATEGY_DICTIONARY[strategy](
        num_partitions=num_partitions, num_coords=num_coords, vals=vals
    )


def get_strategy_ratio_list(ratio_list):
    """
    Takes the a list of ratios, and returns a list of tuples of (strategy,ratio).
    The idea is that for it creates only 1 instance of "all_partition" and "no_partition".
    """
    sr = []
    for strat in STRATEGIES:
        s = strat.__name__
        if s == "all_partition":
            sr.append((s, 1))
        elif s == "no_partition":
            sr.append((s, 0))
        else:
            for r in ratio_list:
                sr.append((s, r))
    return sr


STRATEGIES = [
    all_partition,
    no_partition,
    similar_sized_similar_variance,
    similar_sized_greedy_variances,
    greedy_coordinates_similar_variances,
    similar_sized_random,
]

STRATEGY_DICTIONARY = {name: strat for strat in STRATEGIES for name in strat()}
