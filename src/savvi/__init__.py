from abc import abstractmethod
from numpy.typing import NDArray
from numpy.typing import NDArray
from savvi.special import logbeta, logpow
from savvi.utils import _validate
from typing import List
import cvxpy as cp
import numpy as np


class SequentialTest:
    """
    Parent class for sequential tests.

    Attributes
    ----------
    u : float
        Probability of Type I error.
    n : int
        Sample size.
    odds : float
        Odds in favor of the null hypothesis.
    p : float
        P-value.
    confidence_set : NDArray[np.float64]
        Confidence set for parameters.
    """

    u: float
    n: int
    odds: float
    p: float
    confidence_set: NDArray[np.float64]

    def __init__(self, u: float) -> None:
        _validate(u >= 0 and u <= 1, "u must be in [0, 1]")
        self.u = u
        self.n = 0
        self.odds = 1.0
        self.p = 1.0

    @abstractmethod
    def update(self, x: NDArray, **kwargs) -> None:
        """
        Update test with new data.

        Parameters
        ----------
        x : NDArray
            New data.
        **kwargs
            Keyword arguments for cvxpy.Problem.solve.
        """
        pass

    @abstractmethod
    def update_odds(self, x: NDArray, **kwargs) -> float:
        """
        Update posterior odds.

        Parameters
        ----------
        x : NDArray
            New data.
        **kwargs
            Keyword arguments for cvxpy.Problem.solve.

        Returns
        -------
        float
            Posterior odds.
        """
        pass

    @abstractmethod
    def update_confidence_set(self, **kwargs) -> NDArray[np.float64]:
        """
        Update confidence set.

        Parameters
        ----------
        **kwargs
            Keyword arguments for cvxpy.Problem.solve.

        Returns
        -------
        NDArray[np.float64]
            Confidence set.
        """
        pass


class Multinomial(SequentialTest):
    """
    Implementation of
    [Anytime-Valid Inference for Multinomial Count Data](https://openreview.net/forum?id=a4zg0jiuVi).

    Attributes
    ----------
    theta : cp.Variable
        Multinomial parameter.
    delta : cp.Variable
        Bernoulli or poisson parameter.
    theta_0 : NDArray[np.float64]
        Null Multinomial parameter.
    hypothesis : List[cp.Constraint]
        List of constraints for theta.
    weights : NDArray[np.float64]
        2D array of contrast weights for theta.
    alpha_0 : NDArray[np.float64]
        Prior Dirichlet parameters.
    alpha : NDArray[np.float64]
        Posterior Dirichlet parameters.
    counts : NDArray[np.int64]
        Success counts.
    d : float
        Size of theta.

    Examples
    --------
    {{< include examples/Multinomial.qmd >}}
    """

    theta: cp.Variable
    delta: cp.Variable
    theta_0: NDArray[np.float64]
    hypothesis: List[cp.Constraint]
    alpha_0: NDArray[np.float64]
    alpha: NDArray[np.float64]
    counts: NDArray[np.int64]
    _weights: NDArray[np.float64]

    """
    Initialize a Multinomial test.

    Parameters
    ----------
    u : float
        Probability of Type I error.
    theta_0 : NDArray[np.float64]
        Null Multinomial parameters.
    k : float
        Concentration for Dirichlet prior parameters (`alpha_0 = k * theta_0`).

    Returns
    -------
    Multinomial
        Multinomial test.
    """

    def __init__(
        self,
        u: float,
        theta_0: NDArray[np.float64],
        k: float = 100,
    ) -> None:
        super().__init__(u)
        _validate(
            len(theta_0.shape) == 1, "theta_0 must have a single dimension"
        )
        _validate(theta_0.sum() == 1, "theta_0 must sum to 1")
        self.theta_0 = theta_0
        self.theta = cp.Variable(self.d, name="theta")
        self.delta = cp.Variable(self.d, name="delta")
        self.hypothesis = []
        self.alpha_0 = k * self.theta_0
        self.alpha = k * self.theta_0
        self.counts = np.zeros(self.d, dtype=np.int64)
        self.confidence_set = _initialize_confidence_set(self.d)
        self._weights = np.empty(0)

    @property
    def weights(self) -> NDArray[np.float64]:
        return self._weights

    @weights.setter
    def weights(self, weights: NDArray[np.float64]):
        _validate(self.n == 0, "weights must be set at n = 0")
        message = f"weights must have shape (any, {self.d})"
        _validate(len(weights.shape) == 2, message)
        _validate(weights.shape[1] == self.d, message)
        _validate(weights.sum(axis=None) == 0, "weights must sum to 0")
        self._weights = weights
        self.confidence_set = _initialize_confidence_set(weights.shape[0])

    @property
    def d(self) -> int:
        return self.theta_0.size

    def update(self, x: NDArray[np.int64], **kwargs) -> None:
        shape = (self.d,)
        _validate(x.shape == shape, f"x must have shape {shape}")
        self.n += x.sum()
        self.counts += x
        self.odds = self.update_odds(x, **kwargs)
        self.p = min(self.p, 1 / self.odds)
        self.confidence_set = self.update_confidence_set(**kwargs)
        self.alpha += x

    def update_odds(self, x: NDArray[np.int64], **kwargs) -> float:
        if len(self.hypothesis) == 0:
            log_odds = (
                logbeta(self.alpha + x)
                - logbeta(self.alpha)
                - logpow(self.theta_0, x)
                + np.log(self.odds)
            )
            return np.exp(log_odds)

        if np.isclose(self.p, 0, atol=1e-06):
            return np.inf
        q = cp.Variable()
        objective = cp.Minimize(q)
        c = logbeta(self.alpha_0 + self.counts) - logbeta(self.alpha_0)
        constraints = [
            c
            <= cp.log(q)
            + cp.sum(
                cp.multiply(
                    self.counts,
                    self.delta
                    + np.log(self.theta_0)
                    - cp.log_sum_exp(np.log(self.theta_0) + self.delta),
                )
            ),
        ]
        problem = cp.Problem(objective, constraints + self.hypothesis)
        problem.solve(**kwargs)
        return q.value

    def update_confidence_set(self, **kwargs) -> NDArray[np.float64]:
        if self.weights.size == 0:
            variable = self.theta
            c = logbeta(self.alpha_0 + self.counts) - logbeta(self.alpha_0)
            constraints = [
                c + np.log(self.u)
                <= cp.sum(cp.multiply(self.counts, cp.log(self.theta))),
                cp.sum(self.theta) == 1,
            ]
        else:
            variable = np.array(
                [cp.sum(cp.multiply(w, self.delta)) for w in self.weights]
            )
            c = (
                logbeta(self.alpha_0 + self.counts)
                - logbeta(self.alpha_0)
                + np.log(self.u)
            )
            constraints = [
                c
                <= cp.sum(
                    cp.multiply(
                        self.counts,
                        self.delta
                        + np.log(self.theta_0)
                        - cp.log_sum_exp(np.log(self.theta_0) + self.delta),
                    )
                ),
            ]

        confidence_set = np.empty(self.confidence_set.shape)
        for i in range(self.confidence_set.shape[0]):
            objectives = [cp.Minimize(variable[i]), cp.Maximize(variable[i])]
            for j in range(2):
                problem = cp.Problem(objectives[j], constraints)
                problem.solve(**kwargs)
                value = variable[i].value
                confidence_set[i, j] = value
        confidence_set[:, 0] = np.fmax(
            self.confidence_set[:, 0], confidence_set[:, 0]
        )
        confidence_set[:, 1] = np.fmin(
            self.confidence_set[:, 1], confidence_set[:, 1]
        )

        return confidence_set


def _initialize_confidence_set(d: int):
    confidence_set = np.ones((d, 2))
    confidence_set[:, 0] = confidence_set[:, 0] * np.inf * -1
    confidence_set[:, 1] = confidence_set[:, 1] * np.inf
    return confidence_set
