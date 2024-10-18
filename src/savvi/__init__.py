from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List
import numpy as np


class Inference:
    """
    Base class for inference.
    """

    n: int
    alpha: float
    p_value: np.ndarray
    conf_int: np.ndarray

    def __init__(self, alpha: float, p: int, tests: int = 1):
        """
        Initialize the inference object.
        """
        self.n = 0
        self.alpha = alpha
        self.conf_int = np.full((p, 2), np.inf)
        self.conf_int[:, 0] *= -1
        self.p_value = np.ones(tests)

    @property
    def p(self) -> int:
        """
        Number of parameters.
        """
        return self.conf_int.shape[0]

    @abstractmethod
    def update(self, x: np.ndarray, **kwargs) -> None:
        """
        Update statistics.
        """
        pass

    @abstractmethod
    def calculate_conf_int(self, **kwargs) -> np.ndarray:
        """
        Calculate confidence interval.
        """
        pass

    @abstractmethod
    def calculate_p_value(self, **kwargs) -> float:
        """
        Calculate p-value.
        """
        pass

    def infer(self, **kwargs) -> None:
        """
        Calculate confidence interval and p-value, then:

        * Keep the maximum lower bound and minimum upper bound for the confidence interval;
        * Keep the minimum p-value.
        """
        conf_int = self.calculate_conf_int(**kwargs)
        p_value = self.calculate_p_value(**kwargs)
        self.conf_int[:, 0] = np.fmax(self.conf_int[:, 0], conf_int[:, 0])
        self.conf_int[:, 1] = np.fmin(self.conf_int[:, 1], conf_int[:, 1])
        self.p_value = np.minimum(self.p_value, p_value)

    def batch(self, xs: np.ndarray, **kwargs) -> List[Inference]:
        """
        For each sample unit in the batch: update, infer and keep the inference object.
        """
        sequence = []
        for x in xs:
            self.update(x)
            self.infer(**kwargs)
            sequence.append(deepcopy(self))
        return sequence
