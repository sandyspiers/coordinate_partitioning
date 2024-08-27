import numpy as np


def build_edm(locations, squared=True):
    """
    Uses the Gram matrix and numpy to quickly return the Euclidean distance matrix, squared or not

    Parameters:
     - locations : 2d numpy array with locations are rows
     - squared : whether to return squared EDM or standard EDM
    """
    magnitudes = np.tile(
        np.sum(np.square(locations), axis=1), reps=(locations.shape[0], 1)
    )
    gram = locations.dot(locations.T)
    edm = np.maximum(magnitudes + magnitudes.T - 2 * gram, 0)
    if squared:
        return edm
    else:
        return np.sqrt(edm)


def build_dissaggregated_edm(locations, squared=True):
    """
    Uses the Gram matrix an numpy to quickly return the dissaggregated
    Euclidean distance matrix, squared or not.
    Dissaggreated means it is split by coordinates.
    It returns a matrix of size (n,n,s) for s is the number of coordinates.

    Parameters:
     - locations : 2d numpy array with locations are rows
     - squared : whether to return squared EDM or standard EDM
    """
    locations = np.reshape(locations, (locations.shape[0], 1, locations.shape[1]))
    magnitudes = np.tile(np.square(locations), reps=(1, locations.shape[0], 1))
    gram = np.einsum("ijk,jik->ijk", locations, locations)
    edm = np.maximum(magnitudes + magnitudes.transpose(1, 0, 2) - 2 * gram, 0)
    if squared:
        return edm
    else:
        return np.sqrt(edm)


def partition_dissaggregated_edm(edm, partitions):
    """
    Partitions a dissaggregated EDM and sum over the partitions.
    Returns a (n,n,p) matrix where p is the number of partitions.
    """
    par = np.concatenate(partitions)
    indicies = np.append([0], np.cumsum([len(p) for p in partitions])[:-1])
    return np.add.reduceat(edm[:, :, par.flatten()], indicies, 2)


def gram_recovery(edm, pre_normalize=True, return_evals=False, epsilon=1e-9):
    """
    Returns the set of locations whose squared distance is equal to the distances in input edm.
    The return locations matrix has each location as a row, where first coordinate explains
    most variation, down to the least.
    """
    ## Generate grammian
    n = edm.shape[0]
    mags = np.tile(edm[0], (n, 1))
    gram = (mags + mags.T - edm) / 2

    ## Pre-normalize the grammian
    if pre_normalize:
        gn1 = np.tile(np.sum(gram, axis=0) / n, (n, 1))
        gn2 = gram.sum() / n / n
        gram = gram - gn1 - gn1.T + gn2

    ## Principle root decomposition...
    vals, vecs = np.linalg.eigh(gram)

    ## Get important eigenvals and reorder
    vecs = vecs[:, vals > epsilon][:, ::-1]
    vals = vals[vals > epsilon][::-1]

    ## Determine locations
    locations = vecs.dot(np.diag(np.sqrt(vals)))

    if return_evals:
        return locations, vals
    else:
        return locations
