from savvi import Inference
from typing import List, Tuple
import numpy as np


class LinearRegression(Inference):
    """
    Treatment effect tests from @lindon2024anytimevalidlinearmodelsregression.

    Parameters are calculated using the Recursive Least Squares algorithm.

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

    Parameters
    ----------
    p : int
        Number of covariates.
    alpha : float
        Significance level for inference.
    phi : float, optional
        Prior scale (default is 1).
    """

    p: int
    lamb: float
    n: int
    beta: np.ndarray
    covariance: np.ndarray
    yty: float
    Xty: np.ndarray

    def __init__(self, alpha: float, p: int, phi: float = 1):
        """
        Initialize a Linear Regression model using RLS.

        Parameters
        ----------
        p : int
            Number of covariates.
        alpha : float
            Significance level for inference.
        phi : float, optional
            Prior scale (default is 1).
        lamb : float, optional
            Forgetting factor (default is 1).
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

    # NOTE f-test can't be implemented online
    def infer(self) -> None:
        """
        Perform statistical inference on the coefficients.
        """
        nu = self.n - self.p - 1
        if nu <= 0:
            return

        t2 = self.t_stats() ** 2
        stderrs = self.standard_errors()
        s = self.sigma()
        z2 = (s / stderrs) ** 2

        # Calculate p-values
        p = np.exp(-1 * log_bf(t2, nu, self.phi, z2))

        # Calculate confidence intervals
        r = self.phi / (self.phi + z2)
        radii = stderrs * np.sqrt(
            nu
            * (
                (1 - (r * self.alpha**2) ** (1 / (nu + 1)))
                / np.maximum(0, ((r * self.alpha**2) ** (1 / (nu + 1))) - r)
            )
        )
        lowers = self.beta - radii
        uppers = self.beta + radii
        conf_int = np.column_stack((lowers, uppers))

        # Update inference
        super().intersect(p, conf_int)


def log_bf(t2: np.ndarray, nu: float, phi: float, z2: np.ndarray) -> np.ndarray:
    r = phi / (phi + z2)
    return 0.5 * np.log(r) + (0.5 * (nu + 1)) * (
        np.log(1 + t2 / nu) - np.log(1 + r * t2 / nu)
    )
