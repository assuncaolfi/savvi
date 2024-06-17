from numpy.typing import NDArray
from scipy.special import loggamma, xlogy
import numpy as np


def logbeta(v: NDArray[np.float64]) -> float:
    return loggamma(v).sum() - loggamma(v.sum())


def logpow(v, w) -> float:
    return xlogy(w, v).sum()
