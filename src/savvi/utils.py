from savvi import Inference
from typing import List
import matplotlib.pyplot as plt
import numpy as np


def validate(condition: bool, message: str) -> None:
    if condition is False:
        raise ValueError(message)


def plot(
    sequence: List[Inference],
    truth: np.ndarray | None = None,
    index: List[int] | None = None,
):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    n = [s.n for s in sequence]
    if index is None:
        index = range(sequence[0].p)

    for i in index:
        name = sequence[0].names[i]
        lower = [s.conf_int[i, 0] for s in sequence]
        upper = [s.conf_int[i, 1] for s in sequence]
        ax1.fill_between(n, lower, upper, label=name + " CI", alpha=1 / 3)
        if truth is not None:
            ax1.plot(
                n,
                [truth[i]] * len(n),
                label=name,
                linestyle="dashed",
            )

    first = min(index)
    p_value = [s.p_value[first] for s in sequence]
    ax2.plot(n, p_value, color="black", label="P-value")

    ax1.set_xlabel("n")
    ax1.set_ylabel("Parameter")
    ax2.set_ylabel("P-value")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    return fig, ax1, ax2
