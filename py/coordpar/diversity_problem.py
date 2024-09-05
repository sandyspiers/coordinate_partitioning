import numpy as np

from . import euclid


class DiversityProblem:
    """
    An instance of the Euclidean Max Sum Diversity Problem.
    """

    INSTANCE_TYPES = ["random", "circle"]

    def __init__(self) -> None:
        """
        Initial constructor is empty.
        A generating procedure must be called.
        """
        self.n: int = None
        self.p: int = None
        self.s: int = None
        self.locations: np.ndarray = None
        self.edms: np.ndarray = None

    @classmethod
    def random(cls, n, p, s, axis_limit=100):
        """
        Random diversity problem within a uniform random box
        """
        dp = cls()
        dp._save_nps(n, p, s)
        dp.locations = np.random.rand(n, s) * axis_limit
        dp._build_edm()
        return dp

    @classmethod
    def circle(cls, n, p, s, axis_limit=100):
        """
        Random diversity problem on the edge of a circle
        """
        dp = cls()
        dp._save_nps(n, p, s)
        dp.locations = np.random.normal(size=(n, s))
        dp.locations = (
            dp.locations / np.sqrt(np.sum(np.square(dp.locations), axis=1))[:, None]
        )
        dp.locations = dp.locations * axis_limit / 2
        dp._build_edm()
        return dp

    def _save_nps(self, n, p, s):
        """
        Shorthand save of n,p,s
        """
        self.n = n
        self.p = p
        self.s = s

    def _build_edm(self, squared=False):
        """
        Build the EDM using quick gram method
        """
        self.edm = euclid.build_edm(self.locations, squared)

    def f(self, x):
        return x.dot(self.edm).dot(x) / 2

    def df(self, x):
        return x.dot(self.edm)
