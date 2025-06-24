# File for defining the RQE solution concept
from typing import Callable, Union

from autograd import grad
from autograd import numpy as np

from src.opt import (
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

        self.grad_risk = grad(self.risk_function)
        self.grad_quantal = grad(self.quantal_function)

    def loss_function(
        self, p: np.ndarray, pi, R: np.ndarray, tau: float, epsilon: float
    ) -> tuple[float, float]:
        """
        Compute the loss function for a given policy p and reward matrix R.
        """

        risk_term = R @ pi + self.grad_risk(p, pi) / tau
        quantal_term = R @ p + self.grad_quantal(p) * epsilon
        return risk_term, quantal_term

    def optimize(self, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """
        Optimize the policies for both players using projected gradient descent.
        """
        n1, n2 = R1.shape[0], R2.shape[0]
        pi1 = np.full(n1, 1.0 / n1)
        pi2 = np.full(n2, 1.0 / n2)
        players = np.random.rand(4, 2)
        players /= np.sum(players, axis=1, keepdims=True)

        loss_p1 = lambda p, pi: self.loss_function(p, pi, R1, self.tau1, self.epsilon1)
        loss_p2 = lambda p, pi: self.loss_function(p, pi, R2, self.tau2, self.epsilon2)
        pgd = ProjectedGradientDescent(lr=self.lr, projection=self.projection)

        for _ in range(self.max_iter):
            grads = np.zeros_like(players)

            grads[0], grads[1] = loss_p1(players[0], players[1])
            grads[2], grads[3] = loss_p2(players[2], players[3])
            players = pgd.step(players, grads)

        return players

    @staticmethod
    def print_game(R1: np.ndarray, R2: np.ndarray):
        """
        Print the game matrices for both players.
        """
        for i in range(R1.shape[0]):
            row = []
            for j in range(R1.shape[1]):
                row.append(f"{int(R1[i, j])}, {int(R2[i, j])}")
            print(" | ".join(row))


if __name__ == "__main__":
    # Example usage
    R1 = np.array([[3, 0], [5, 1]])
    R2 = np.array([[3, 5], [0, 1]])
    RQE.print_game(R1, R2)
    rqe_solver = RQE(tau1=0.005, tau2=0.005, epsilon1=200, epsilon2=200)
    pi1, pi2 = rqe_solver.optimize(R1, R2)

    print("Computed policies for RQE")
    # print("Player 1:", np.argmax(pi1), "with policy:", pi1)
    print(f"Player 1: Best Action {np.argmax(pi1)} with policy: {pi1}")
    # print("Player 2:", np.argmax(pi2), "with policy:", pi2)
    print(f"Player 2: Best Action {np.argmax(pi2)} with policy: {pi2}")

    import nashpy as nash

    game = nash.Game(R1, R2)
    equilibrium = game.support_enumeration()
    print("Nash Equilibrium using nashpy:")

    for eq in equilibrium:
        print(f"Player 1: Best Action {np.argmax(eq[0])} with policy: {eq[0]}")
        print(f"Player 2: Best Action {np.argmax(eq[1])} with policy: {eq[1]}")
