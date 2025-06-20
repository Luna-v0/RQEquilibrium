from typing import Callable

from autograd import grad
from autograd import numpy as np
from scipy.special import logsumexp


class ProjectedGradientDescent:
    """
    A class for performing projected gradient descent to solve optimization problems.


    """

    def __init__(
        self,
        cost_function: Callable,
        lr: float = 0.1,
        epochs=1000,
        projection=lambda x: x,
    ):

        self.cost_function = cost_function
        self.lr = lr
        self.epochs = epochs
        self.projection = projection

    def optimize(self, initial_points) -> np.array:
        """
        Perform the optimization using projected gradient descent.

        Returns:
            np.array: The optimized points after the specified number of epochs.
        """
        gradient = grad(self.cost_function)

        w = np.array(initial_points, dtype=float)

        for _ in range(self.epochs):
            w = self.projection(w - self.lr * gradient(w))

        return w


def project_simplex(x: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the simplex defined by the constraints that all elements are non-negative
    and sum to 1.
    """
    if x.ndim != 1:
        return np.vstack([project_simplex(xi) for xi in x])

    if np.all(x >= 0) and np.isclose(np.sum(x), 1):
        return x

    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / np.arange(1, n + 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0)


def kl_divergence(p: np.ndarray, q: np.ndarray, tau: float) -> float:
    """
    Compute the KL divergence between two probability distributions p and q.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return np.sum(p * (np.log(p) - np.log(q))) / tau


def kl_reversed(p: np.ndarray, q: np.ndarray, tau: float) -> float:
    """
    Compute the reversed KL divergence between two probability distributions p and q.
    """

    return kl_divergence(q, p, tau)


def negative_entropy(p: np.ndarray) -> float:
    """
    Compute the negative entropy of a probability distribution p.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.sum(p * np.log(p))


def log_barrier(p: np.ndarray) -> float:
    """
    Compute the log barrier function for a probability distribution p.
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.sum(np.log(p))
