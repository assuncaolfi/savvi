from collections.abc import Callable
from savvi import Inference
from savvi.special import logbeta, logpow
from savvi.utils import validate
from typing import List
import cvxpy as cp
import numpy as np


class Multinomial(Inference):
    """
    Sample ratio mismatch test from @lindon2022anytimevalid.

    * Parent class: [`Inference`](/reference/Inference.qmd)
    * [Example](../examples/Multinomial.qmd)

    Parameters
    ----------
    alpha : float
        Probability of Type I error $\\alpha$.
    theta_0 : np.ndarray
        Null Multinomial parameters $\\mathbf{\\theta}_0$.
    k : float
        Concentration for Dirichlet prior parameters $\\mathbf{\\alpha}_0 = k \\mathbf{\\theta}_0$.
    """

    theta_0: np.ndarray
    """Null Multinomial parameters $\\mathbf{\\theta}_0$."""
    alpha_0: np.ndarray
    """Prior Dirichlet parameters $\\mathbf{\\alpha}_0$."""
    counts: np.ndarray
    """Success counts."""
    odds: float
    """Posterior odds."""

    def __init__(
        self,
        alpha: float,
        theta_0: np.ndarray,
        k: float = 100,
    ) -> None:
        super().__init__(alpha, len(theta_0))
        validate(len(theta_0.shape) == 1, "theta_0 must have a single dimension")
        validate(theta_0.sum() == 1, "theta_0 must sum to 1")
        self.theta_0 = theta_0
        self.alpha_0 = k * self.theta_0
        self.counts = np.zeros(self.d, dtype=np.int64)
        self.odds = 1.0

    @property
    def names(self) -> List[str]:
        return [f"$\\theta_{i}$" for i in range(self.d)]

    @property
    def estimate(self) -> np.ndarray:
        return self.theta

    @property
    def d(self) -> int:
        return self.theta_0.size

    @property
    def alpha_n(self) -> np.ndarray:
        return self.alpha_0 + self.counts

    @property
    def theta(self) -> np.ndarray:
        """
        Estimate of theta $\\mathbf{\\hat{\\theta}}$.

        Returns
        -------
        :
        """
        return self.counts / self.counts.sum()

    def update(self, x: np.ndarray) -> None:
        """
        Update the model with success counts.

        Parameters
        ----------
        x : np.ndarray
            Success counts.
        """
        self.n += x.sum()
        self.counts += x
        self.odds = self.update_odds(x)

    def update_odds(self, x: np.ndarray) -> float:
        log_odds = (
            logbeta(self.alpha_n)
            - logbeta(self.alpha_n - x)
            - logpow(self.theta_0, x)
            + np.log(self.odds)
        )
        return np.exp(log_odds)

    def calculate_conf_int(self, **kwargs) -> np.ndarray:
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
        return conf_int

    def calculate_p_value(self, **kwargs) -> float:
        return 1 / self.odds


class InhomogeneousBernoulliProcess(Multinomial):
    """
    Conversion rate optimization test from @lindon2022anytimevalid.

    * Parent class: [`Multinomial`](/reference/Multinomial.qmd)
    * [Example](../examples/InhomogeneousBernoulliProcess.qmd)

    Parameters
    ----------
    alpha : float
        Probability of Type I error $\\alpha$.
    rho : np.ndarray
        Assignment probabilities $\\mathbf{\\rho}$.
    hypothesis : Callable[[cp.Variable], List[cp.Constraint]]
        Function to generate hypothesis constraints.
    weights : np.ndarray
        Contrast weights $W$.
    k : float
        Concentration for Dirichlet prior parameters $\\mathbf{\\alpha}_0 = k \\mathbf{\\rho}$.
    """

    hypothesis: Callable[[cp.Variable], List[cp.Constraint]]
    """Function to generate hypothesis constraints."""
    weights: np.ndarray
    """Contrast weights $W$."""

    def __init__(
        self,
        alpha: float,
        rho: np.ndarray,
        hypothesis: Callable[[cp.Variable], List[cp.Constraint]],
        weights: np.ndarray,
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
    def names(self) -> List[str]:
        names = []
        for weight in self.weights:
            numbers = ["" if w == 1 else "-" if w == -1 else f"{w}" for w in weight]
            name = [f"{n} \\delta_{i}" for i, n in enumerate(numbers) if n != "0"]
            names.append(" + ".join(name))
        return names

    @property
    def estimate(self) -> np.ndarray:
        return self.contrasts

    @property
    def contrasts(self) -> np.ndarray:
        """
        Estimate of contrasts $\\hat{W \\mathbf{\\delta}}$.

        Returns
        -------
        :
        """
        delta = self.theta / self.theta_0
        return self.weights @ np.log(delta, where=delta > 0)

    def update_odds(self, x: np.ndarray, **kwargs) -> float:
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

    def calculate_conf_int(self, **kwargs) -> np.ndarray:
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

    * Parent class: [`InhomogeneousBernoulliProcess`](/reference/InhomogeneousBernoulliProcess.qmd)
    * [Example](../examples/InhomogeneousPoissonProcess.qmd)

    Parameters
    ----------
    alpha : float
        Probability of Type I error $\\alpha$.
    rho : np.ndarray
        Assignment probabilities $\\mathbf{\\rho}$.
    weights : np.ndarray
        Contrast weights $W$.
    k : float
        Concentration for Dirichlet prior parameters $\\mathbf{\\alpha}_0 = k \\mathbf{\\rho}$.
    """

    def __init__(
        self,
        alpha: float,
        rho: np.ndarray,
        weights: np.ndarray,
        k: float = 100,
    ) -> None:
        hypothesis = lambda _: []
        super().__init__(alpha, rho, hypothesis, weights, k)

    def update_odds(self, x: np.ndarray, **kwargs) -> float:
        return Multinomial.update_odds(self, x, **kwargs)
