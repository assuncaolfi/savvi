from savvi import Inference
from typing import List, Tuple
import numpy as np


# NOTE f-test can't be implemented online
class LinearRegression(Inference):
    """
    Covariate adjusted t-tests from @lindon2024anytimevalidlinearmodelsregression.

    Coefficients and covariance matrix are calculated using the Recursive Least
    Squares algorithm.

    * Parent class: [`Inference`](/reference/Inference.qmd)
    * [Example](../examples/LinearRegression.qmd)

    Parameters
    ----------
    alpha : float
        Probability of Type I error $\\alpha$.
    p : int
        Number of covariates $p$.
    phi : float, optional
        Prior scale $\\phi$.
    """

    phi: float
    """Prior scale $\\phi$."""
    beta: np.ndarray
    """Estimate of regression coefficients $\\hat{\\beta}$."""
    covariance: np.ndarray
    """Estimate of covariance matrix $\\hat{\\Sigma}$."""
    yty: float
    """Sum of squared response values $y^T y$."""
    Xty: np.ndarray
    """Sum of products of covariates and response $X^T y$."""

    def __init__(self, alpha: float, p: int, phi: float = 1):
        self.lamb = 1  # TODO remove
        self.phi = phi
        self.beta = np.zeros(p)
        self.covariance = np.eye(p) * 1e6
        self.yty = 0.0
        self.Xty = np.zeros(p)
        super().__init__(alpha, p, tests=p)

    @property
    def names(self) -> List[str]:
        return [f"$\\beta_{i}$" for i in range(self.p)]

    @property
    def estimate(self) -> np.ndarray:
        return self.beta

    def update(self, yx: np.ndarray) -> None:
        """
        Update the model with new data.

        Parameters
        ----------
        yx : np.ndarray
            Array of response and covariate values $[y, x_1, \dots, x_p]$.
        """
        self.n += 1

        y = yx[0]
        x = yx[1:]
        self.yty = self.lamb * self.yty + y * y
        self.Xty = self.lamb * self.Xty + y * x

        error = y - self.predict(x)
        Px = np.dot(self.covariance, x)
        k = Px / (self.lamb + np.dot(x, Px))
        self.beta += k * error
        self.covariance = (
            1 / self.lamb * (self.covariance - np.outer(k, np.dot(x, self.covariance)))
        )
        self.covariance = (self.covariance + self.covariance.T) / 2  # Ensure symmetry

    def calculate_conf_int(self) -> np.ndarray:
        if self.nu() <= 0:
            return self.conf_int
        stderrs = self.standard_errors()
        r = self.phi / (self.phi + self.z2())
        radii = stderrs * np.sqrt(
            self.nu()
            * (
                (1 - (r * self.alpha**2) ** (1 / (self.nu() + 1)))
                / np.maximum(0, ((r * self.alpha**2) ** (1 / (self.nu() + 1))) - r)
            )
        )
        lowers = self.beta - radii
        uppers = self.beta + radii
        return np.column_stack((lowers, uppers))

    def calculate_p_value(self) -> np.ndarray:
        if self.nu() <= 0:
            return self.p_value
        t2 = self.t_stats() ** 2
        return np.exp(-1 * log_bf(t2, self.nu(), self.phi, self.z2()))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for given covariates.

        Parameters
        ----------
        X : np.ndarray
            Matrix of covariates.

        Returns
        -------
        :
        """
        return np.dot(X, self.beta)

    def sse(self) -> float:
        """
        Compute the Sum of Squared Errors (SSE).

        Returns
        -------
        :
        """
        return self.yty - np.dot(self.beta, self.Xty)

    def sigma(self) -> float:
        """
        Estimate the standard deviation of the error term.

        Returns
        -------
        :
        """
        return np.sqrt(self.sse() / (self.n - self.p))

    def standard_errors(self) -> np.ndarray:
        """
        Estimate the standard errors of the coefficients.

        Returns
        -------
        :
        """
        return np.sqrt(np.diag(self.covariance) * self.sigma() ** 2)

    def t_stats(self) -> np.ndarray:
        """
        Calculate the t statistics of the coefficients.

        Returns
        -------
        :
        """
        return self.beta / self.standard_errors()

    def nu(self) -> int:
        """
        Degrees of freedom.

        Returns
        -------
        :
        """
        return self.n - self.p - 1

    def z2(self) -> np.ndarray:
        """
        Calculate the squared z-scores.

        Returns
        -------
        :
        """
        stderrs = self.standard_errors()
        s = self.sigma()
        return (s / stderrs) ** 2


def log_bf(t2: np.ndarray, nu: float, phi: float, z2: np.ndarray) -> np.ndarray:
    r = phi / (phi + z2)
    return 0.5 * np.log(r) + (0.5 * (nu + 1)) * (
        np.log(1 + t2 / nu) - np.log(1 + r * t2 / nu)
    )
