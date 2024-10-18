from collections.abc import Callable
from numpy.typing import NDArray
from savvi import Inference
from savvi.special import logbeta, logpow
from savvi.utils import validate
from typing import List
import cvxpy as cp
import numpy as np


class Multinomial(Inference):
    """
    Sample ratio mismatch test from @lindon2022anytimevalid.
    See [example](../examples/Multinomial.qmd).

    Attributes
    ----------
    theta : cp.Variable
        Multinomial parameter.
    theta_0 : NDArray[np.float64]
        Null Multinomial parameter.
    alpha_0 : NDArray[np.float64]
        Prior Dirichlet parameters.
    alpha : NDArray[np.float64]
        Posterior Dirichlet parameters.
    counts : NDArray[np.int64]
        Success counts.
    d : float
        Size of theta.
    """

    def __init__(
        self,
        alpha: float,
        theta_0: NDArray[np.float64],
        k: float = 100,
    ) -> None:
        """
        Initialize a Multinomial test.

        Parameters
        ----------
        alpha : float
            Probability of Type I error.
        theta_0 : NDArray[np.float64]
            Null Multinomial parameters.
        k : float
            Concentration for Dirichlet prior parameters (`alpha_0 = k * theta_0`).
        """
        super().__init__(alpha, len(theta_0))
        validate(len(theta_0.shape) == 1, "theta_0 must have a single dimension")
        validate(theta_0.sum() == 1, "theta_0 must sum to 1")
        self.theta_0 = theta_0
        self.alpha_0 = k * self.theta_0
        self.counts = np.zeros(self.d, dtype=np.int64)
        self.odds = 1

    @property
    def d(self) -> int:
        return self.theta_0.size

    @property
    def names(self) -> List[str]:
        return [f"$\\theta_{i}$" for i in range(self.d)]

    @property
    def alpha_n(self) -> NDArray[np.float64]:
        return self.alpha_0 + self.counts

    @property
    def theta(self) -> NDArray[np.float64]:
        return self.counts / self.counts.sum()

    def update(self, x: NDArray[np.int64]) -> None:
        self.n += x.sum()
        self.counts += x
        self.odds = self.update_odds(x)

    def update_odds(self, x: NDArray[np.int64]) -> float:
        log_odds = (
            logbeta(self.alpha_n)
            - logbeta(self.alpha_n - x)
            - logpow(self.theta_0, x)
            + np.log(self.odds)
        )
        return np.exp(log_odds)

    def calculate_conf_int(self, **kwargs) -> NDArray[np.float64]:
        theta = cp.Variable(self.d, name="theta")
        c = logbeta(self.alpha_n) - logbeta(self.alpha_0)
        constraints = [
            c + np.log(self.alpha) <= cp.sum(cp.multiply(self.counts, cp.log(theta))),
            cp.sum(theta) == 1,
        ]
        conf_int = np.empty((self.d, 2))
        for i in range(self.d):
            objectives = [cp.Minimize(theta[i]), cp.Maximize(theta[i])]
            for j in range(2):
                problem = cp.Problem(objectives[j], constraints)
                problem.solve(**kwargs)
                value = theta[i].value
                conf_int[i, j] = value
        self.conf_int = conf_int
        return conf_int

    def calculate_p_value(self, **kwargs) -> float:
        return 1 / self.odds


class InhomogeneousBernoulliProcess(Multinomial):
    """
    Conversion rate optimization test from @lindon2022anytimevalid.
    See [example](../examples/InhomogeneousBernoulliProcess.qmd).
    """

    def __init__(
        self,
        alpha: float,
        rho: NDArray[np.float64],
        hypothesis: Callable[[cp.Variable], List[cp.Constraint]],
        weights: NDArray[np.float64],
        k: float = 100,
    ) -> None:
        super().__init__(alpha, rho, k)
        Inference.__init__(self, alpha, weights.shape[0])
        message = f"weights must have shape (any, {self.d})"
        validate(len(weights.shape) == 2, message)
        validate(weights.shape[1] == self.d, message)
        validate(weights.sum(axis=None) == 0, "weights must sum to 0")
        self.hypothesis = hypothesis
        self.weights = weights

    @property
    def delta(self) -> NDArray[np.float64]:
        contrast = self.theta / self.theta_0
        return self.weights @ np.log(contrast, where=contrast > 0)

    @property
    def names(self) -> List[str]:
        names = []
        for weight in self.weights:
            numbers = ["" if w == 1 else "-" if w == -1 else f"{w}" for w in weight]
            name = [f"${n} \\delta_{i}$" for i, n in enumerate(numbers) if n != "0"]
            names.append(" + ".join(name))
        return names

    def update_odds(self, x: NDArray[np.int64], **kwargs) -> float:
        if np.isclose(self.p_value, 0, atol=1e-06):
            return np.inf
        q = cp.Variable()
        objective = cp.Minimize(q)
        c = logbeta(self.alpha_n) - logbeta(self.alpha_0)
        delta = cp.Variable(self.d, name="delta")
        constraints = [
            c
            <= cp.log(q)
            + cp.sum(
                cp.multiply(
                    self.counts,
                    delta
                    + np.log(self.theta_0)
                    - cp.log_sum_exp(np.log(self.theta_0) + delta),
                )
            ),
        ]
        hypothesis = self.hypothesis(delta)
        problem = cp.Problem(objective, constraints + hypothesis)
        problem.solve(**kwargs)
        return q.value

    def calculate_conf_int(self, **kwargs) -> NDArray[np.float64]:
        c = logbeta(self.alpha_n) - logbeta(self.alpha_0) + np.log(self.alpha)
        delta = cp.Variable(self.d, name="delta")
        constraints = [
            c
            <= cp.sum(
                cp.multiply(
                    self.counts,
                    delta
                    + np.log(self.theta_0)
                    - cp.log_sum_exp(np.log(self.theta_0) + delta),
                )
            ),
        ]
        conf_int = np.empty(self.conf_int.shape)
        for i in range(self.conf_int.shape[0]):
            contrast = cp.sum(cp.multiply(self.weights[i], delta))
            objectives = [cp.Minimize(contrast), cp.Maximize(contrast)]
            for j in range(self.conf_int.shape[1]):
                problem = cp.Problem(objectives[j], constraints)
                problem.solve(**kwargs)
                conf_int[i, j] = contrast.value
        return conf_int


class InhomogeneousPoissonProcess(InhomogeneousBernoulliProcess):
    """
    Canary software release test from @lindon2022anytimevalid.
    See [example](../examples/InhomogeneousPoissonProcess.qmd).
    """

    def __init__(
        self,
        alpha: float,
        rho: NDArray[np.float64],
        weights: NDArray[np.float64],
        k: float = 100,
    ) -> None:
        hypothesis = lambda _: []
        super().__init__(alpha, rho, hypothesis, weights, k)

    def update_odds(self, x: NDArray[np.int64], **kwargs) -> float:
        return Multinomial.update_odds(self, x, **kwargs)
