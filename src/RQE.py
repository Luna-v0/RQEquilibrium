# File for defining the RQE solution concept
from typing import Callable, Union

import numpy as np

from opt import (
    ProjectedGradientDescent,
    kl_divergence,
    kl_reversed,
    log_barrier,
    negative_entropy,
    project_simplex,
)


class RQE:
    quantal_function: Callable
    risk_function: Callable

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
        projection: Callable = project_simplex,
    ):
        self.tau1 = tau1
        self.tau2 = tau2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.lr = lr
        self.max_iter = max_iter
        self.br_iters = br_iters
        self.EPS = 1e-12
        self.projection = projection

        if hasattr(quantal_function, "__call__"):
            self.quantal_function = quantal_function
        elif quantal_function == "negative_entropy":
            self.quantal_function = negative_entropy
        elif quantal_function == "log_barrier":
            self.quantal_function = log_barrier
        else:
            raise ValueError("Invalid quantal function specified.")

        if hasattr(risk_function, "__call__"):
            self.risk_function = risk_function
        elif risk_function == "kl_divergence":
            self.risk_function = kl_divergence
        elif risk_function == "kl_reversed":
            self.risk_function = kl_reversed

    def loss_function(
        self, p: np.ndarray, pi, R: np.ndarray, tau: float, epsilon: float
    ) -> float:
        """
        Compute the loss function for a given policy p and reward matrix R.
        """
        expected_reward = -p @ R @ pi
        risk_term = self.risk_function(p, pi, tau)
        quantal_term = self.quantal_function(p) * epsilon
        return expected_reward + risk_term + quantal_term + epsilon * log_barrier(p)

    def optimize(self, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """
        Optimize the policies for both players using projected gradient descent.
        """
        n1, n2 = R1.shape[0], R2.shape[0]
        pi1 = np.full(n1, 1.0 / n1)
        pi2 = np.full(n2, 1.0 / n2)

        loss_p1 = lambda p: self.loss_function(p, pi2, R1, self.tau1, self.epsilon1)
        loss_p2 = lambda p: self.loss_function(p, pi1, R2, self.tau2, self.epsilon2)
        pgd1 = ProjectedGradientDescent(
            cost_function=loss_p1,
            lr=self.lr,
            epochs=self.max_iter,
            projection=self.projection,
        )
        pgd2 = ProjectedGradientDescent(
            cost_function=loss_p2,
            lr=self.lr,
            epochs=self.max_iter,
            projection=self.projection,
        )

        for _ in range(self.max_iter):
            old_pi1 = pi1.copy()
            old_pi2 = pi2.copy()

            pi1 = pgd1.optimize(pi1)
            pi2 = pgd2.optimize(pi2)

            # Update policies based on the current policies
            pi1 = self.projection(pi1)
            pi2 = self.projection(pi2)

            # Check for convergence
            if (
                np.linalg.norm(pi1 - old_pi1) < self.EPS
                and np.linalg.norm(pi2 - old_pi2) < self.EPS
            ):
                break

        return pi1, pi2


if __name__ == "__main__":
    # Example usage
    R1 = np.array([[-6, -6], [0, -10]])
    R2 = np.array([[-10, 0], [-1, -1]])

    rqe_solver = RQE(tau1=0.1, tau2=0.1, epsilon1=1, epsilon2=1)
    pi1, pi2 = rqe_solver.optimize(R1, R2)

    print("Computed policies:")
    print("Player 1:", pi1)
    print("Player 2:", pi2)
