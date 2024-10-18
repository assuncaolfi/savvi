from savvi import Inference
from typing import List, Tuple
import numpy as np


# NOTE f-test can't be implemented online
class LinearRegression(Inference):
    """
    Treatment effect tests from @lindon2024anytimevalidlinearmodelsregression.

    Coefficients are calculated using the Recursive Least Squares algorithm.

    See [example](../examples/linear_regression.qmd).

    Attributes
    ----------
    beta : np.ndarray
        Estimate of regression coefficients.
    covariance : np.ndarray
        Estimate of covariance matrix.
    yty : float
        Sum of squared response values.
    Xty : np.ndarray
        Sum of products of covariates and response.
    """

    beta: np.ndarray
    covariance: np.ndarray
    yty: float
    Xty: np.ndarray

    def __init__(self, alpha: float, p: int, phi: float = 1):
        """
        Initialize a Linear Regression model using RLS.

        Parameters
        ----------
        alpha : float
            Significance level for inference.
        p : int
            Number of covariates.
        phi : float, optional
            Prior scale (default is 1).
        """
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

    def update(self, xy: Tuple[np.ndarray, float]) -> None:
        """
        Update the model with new data.

        Parameters
        ----------
        xy : Tuple[np.ndarray, float]
            Tuple of response and covariate values.
        """
        self.n += 1

        x, y = xy
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
        """
        Calculate confidence intervals for the coefficients.

        Returns
        -------
        np.ndarray
            Confidence intervals for the coefficients.
        """
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
        """
        Calculate p-values for the coefficients.

        Returns
        -------
        np.ndarray
            P-values for the coefficients.
        """
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
        np.ndarray
            Predicted values.
        """
        return np.dot(X, self.beta)

    def sse(self) -> float:
        """
        Compute the Sum of Squared Errors (SSE).

        Returns
        -------
        float
            SSE value.
        """
        return self.yty - np.dot(self.beta, self.Xty)

    def sigma(self) -> float:
        """
        Estimate the standard deviation of the error term.

        Returns
        -------
        float
            Estimate of the standard deviation of the error term.
        """
        return np.sqrt(self.sse() / (self.n - self.p))

    def standard_errors(self) -> np.ndarray:
        """
        Estimate the standard errors of the coefficients.

        Returns
        -------
        np.ndarray
            Estimate of the standard errors of the coefficients.
        """
        return np.sqrt(np.diag(self.covariance) * self.sigma() ** 2)

    def t_stats(self) -> np.ndarray:
        """
        Calculate the t statistics of the coefficients.

        Returns
        -------
        np.ndarray
            T statistics of the coefficients.
        """
        return self.beta / self.standard_errors()

    def nu(self) -> int:
        """
        Degrees of freedom.

        Returns
        -------
        int
            Degrees of freedom.
        """
        return self.n - self.p - 1

    def z2(self) -> np.ndarray:
        """
        Calculate the squared z-scores.

        Returns
        -------
        np.ndarray
            Squared z-scores.
        """
        stderrs = self.standard_errors()
        s = self.sigma()
        return (s / stderrs) ** 2


def log_bf(t2: np.ndarray, nu: float, phi: float, z2: np.ndarray) -> np.ndarray:
    r = phi / (phi + z2)
    return 0.5 * np.log(r) + (0.5 * (nu + 1)) * (
        np.log(1 + t2 / nu) - np.log(1 + r * t2 / nu)
    )
