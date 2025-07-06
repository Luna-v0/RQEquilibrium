from typing import Callable

from autograd import grad
from autograd import numpy as np


class ProjectedGradientDescent:
    """
    A class for performing projected gradient descent to solve optimization problems.

    Attributes:
        lr (float): Learning rate for the gradient descent step.
        projection (Callable): Function to project the point onto a feasible set after each step.

    Methods:
        step(w: np.ndarray, gradients_values: np.ndarray) -> np.ndarray:
            Performs a single step of projected gradient descent.
            Updates the current point `w` by subtracting the scaled gradients and projecting it onto the feasible set.
    """

    def __init__(
        self,
        lr: float = 0.1,
        projection=lambda x: x,
    ):

        self.lr = lr
        self.projection = projection

    def step(self, w: np.ndarray, gradients_values: np.ndarray) -> np.ndarray:
        """
        Perform a single step of projected gradient descent.

        Parameters:
            w (np.ndarray): The current point.
            gradients_values (np.ndarray): The computed gradients at the current point.
        Returns:
            np.ndarray: The updated point after one step.
        """

        w -= self.lr * gradients_values
        return self.projection(np.clip(w, 1e-12, 1 - 1e-12))


def project_simplex(v, s=1):
    """
    Projects a vector v onto the simplex defined by the sum of its components being equal to s.
    The simplex is defined as the set of vectors x such that:
    x >= 0 and sum(x) = s.

    Parameters:
        v (np.ndarray): The vector to be projected.
        s (float): The sum that the components of the projected vector should equal. Default is 1.
    Returns:
        np.ndarray: The projected vector onto the simplex.
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    v = np.reshape(v, (v.shape[0]))
    (n,) = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.all(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho + 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the KL divergence between two probability distributions p and q.

    Parameters:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.
    Returns:
        float: The KL divergence between p and q.
    """
    if q.ndim > 1:
        return np.sum([kl_divergence(p, q_i) for q_i in q])

    return np.sum(p * (np.log(p) - np.log(q)))


def kl_reversed(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the reversed KL divergence between two probability distributions p and q.

    Parameters:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.
    Returns:
        float: The reversed KL divergence between p and q.
    """

    return kl_divergence(q, p)


def negative_entropy(p: np.ndarray) -> float:
    """
    Compute the negative entropy of a probability distribution p.
    Parameters:
        p (np.ndarray): The probability distribution.
    Returns:
        float: The negative entropy of the distribution.
    """
    return -np.sum(p * np.log(p))


def log_barrier(p: np.ndarray) -> float:
    """
    Compute the log barrier function for a probability distribution p.
    Parameters:
        p (np.ndarray): The probability distribution.
    Returns:
        float: The log barrier value for the distribution.
    """
    return -np.sum(np.log(p))
