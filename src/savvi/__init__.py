from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from py_markdown_table.markdown_table import markdown_table
from typing import List
import numpy as np


class Inference:
    """
    Base class for inference.

    Parameters
    ----------
    alpha : float
        Probability of Type I error $\\alpha$.
    p : int
        Number of parameters $p$.
    tests : int
        Number of hypothesis tests.
    """

    alpha: float
    """Probability of Type I error $\\alpha$."""
    n: int
    """Number of observations $n$."""
    conf_int: np.ndarray
    """Confidence intervals for each parameter ($p \\times 2$ matrix)."""
    p_value: np.ndarray
    """P-values for each hypothesis test."""

    def __init__(self, alpha: float, p: int, tests: int = 1):
        self.n = 0
        self.alpha = alpha
        self.conf_int = np.full((p, 2), np.inf)
        self.conf_int[:, 0] *= -1
        self.p_value = np.ones(tests)

    @property
    def p(self) -> int:
        """
        Number of parameters $p$.
        """
        return self.conf_int.shape[0]

    @property
    @abstractmethod
    def names(self) -> List[str]:
        """
        Names for each parameter.
        """
        pass

    @property
    @abstractmethod
    def estimate(self) -> np.ndarray:
        """
        Estimates for each parameter.
        """
        pass

    def batch(self, xs: np.ndarray, **kwargs) -> List[Inference]:
        """
        For each sample unit in the batch

        1) call `update`,
        2) call `infer`, and
        3) append the `Inference` object to a list.

        Returns
        -------
        :
        """
        sequence = []
        for x in xs:
            self.update(x)
            self.infer(**kwargs)
            sequence.append(deepcopy(self))
        return sequence

    @abstractmethod
    def update(self, x: np.ndarray, **kwargs) -> None:
        """
        Update statistics with new data.
        """
        pass

    def infer(self, **kwargs) -> None:
        """
        Calculate confidence interval and p-value, then

        1) keep the maximum lower bound and minimum upper bound for the
        confidence interval; and
        2) keep the minimum p-value.
        """
        conf_int = self.calculate_conf_int(**kwargs)
        p_value = self.calculate_p_value(**kwargs)
        self.conf_int[:, 0] = np.fmax(self.conf_int[:, 0], conf_int[:, 0])
        self.conf_int[:, 1] = np.fmin(self.conf_int[:, 1], conf_int[:, 1])
        self.p_value = np.minimum(self.p_value, p_value)

    @abstractmethod
    def calculate_conf_int(self, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def calculate_p_value(self, **kwargs) -> float:
        pass

    def __str__(self) -> str:
        is_multiple = self.p_value.size > 1
        data = []
        for p in range(self.p):
            entry = {
                "Parameter": self.names[p],
                "Estimate": self.estimate[p],
                "CI Lower": self.conf_int[p, 0],
                "CI Upper": self.conf_int[p, 1],
            }
            if is_multiple:
                entry["P-value"] = self.p_value[p]
            data.append(entry)
        table = (
            markdown_table(data)
            .set_params(float_rounding=4, quote=False, row_sep="markdown")
            .get_markdown()
        )
        caption = f"\nSample size: {self.n}"
        if not is_multiple:
            caption += f", P-value: {self.p_value[0]:.4f}"
        return table + caption

    def _repr_markdown_(self) -> str:
        return self.__str__()
