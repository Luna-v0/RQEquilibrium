# File for defining the RQE solution concept
from typing import Callable, Union

import numpy as np

from opt import (
    ProjectedGradientDescent,
    kl_divergence,
    kl_reversed,
    log_barrier,
    negative_entropy,
)


class RQE:
    def __init__(
        self,
        tau1: float,
        tau2: float,
        epsilon1: float,
        epsilon2: float,
        lr=0.1,
        max_iter=500,
        br_iters=50,
        quantal_function: Union[Callable, str] = "negative_entropy",
        risk_function: Union[Callable, str] = "kl_divergence",
    ):
        self.tau1 = tau1
        self.tau2 = tau2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.lr = lr
        self.max_iter = max_iter
        self.br_iters = br_iters
        self.EPS = 1e-12

        if type(quantal_function) is not str:
            self.quantal_function = quantal_function
        elif quantal_function == "negative_entropy":
            self.quantal_function = negative_entropy
        elif quantal_function == "log_barrier":
            self.quantal_function = log_barrier
        else:
            raise ValueError("Invalid quantal function specified.")

        if type(risk_function) is not str:
            self.risk_function = risk_function
        elif risk_function == "kl_divergence":
            self.risk_function = kl_divergence
        elif risk_function == "kl_reversed":
            self.risk_function = kl_reversed

        self.gradient_descent = ProjectedGradientDescent

    def optimize(self, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """
        Optimize the policies for both players using projected gradient descent.
        """
        pass


if __name__ == "__main__":
    # Example usage
    R1 = np.array([[1, -1], [-1, 1]])
    R2 = np.array([[-1, 1], [1, -1]])

    rqe_solver = RQE(tau1=1.0, tau2=1.0, epsilon1=0.01, epsilon2=0.01)
    pi1, pi2 = rqe_solver.optimize(R1, R2)

    print("Computed policies:")
    print("Player 1:", pi1)
    print("Player 2:", pi2)
