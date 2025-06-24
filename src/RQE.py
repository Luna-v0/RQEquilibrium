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
        self,
        p: np.ndarray,
        pi1: np.ndarray,
        pi2: np.ndarray,
        R: np.ndarray,
        tau: float,
        epsilon: float,
    ) -> tuple[float, float]:
        """
        Compute the loss function for a given policy p and reward matrix R.
        """

        risk_term = R.T @ p + (1 / tau) * self.grad_risk(pi1, pi2)
        quantal_term = -R @ pi1 + epsilon * self.grad_quantal(p)
        return risk_term, quantal_term

    def optimize(self, R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
        """
        Optimize the policies for both players using projected gradient descent.
        """
        R2 = R2.T  # Ensure R2 is transposed to match the expected shape
        players = np.random.rand(4, 2)
        players /= np.sum(players, axis=1, keepdims=True)

        loss_p1 = lambda p, pi1, pi2: self.loss_function(
            p, pi1, pi2, R1, self.tau1, self.epsilon1
        )
        loss_p2 = lambda p, pi1, pi2: self.loss_function(
            p, pi1, pi2, R2, self.tau2, self.epsilon2
        )
        pgd = ProjectedGradientDescent(
            lr=self.lr,
            projection=self.projection,
        )

        for _ in range(self.max_iter):
            grads = np.zeros_like(players)

            grads[1], grads[0] = loss_p1(players[0], players[1], players[2])
            grads[3], grads[2] = loss_p2(players[2], players[3], players[0])

            players = pgd.step(players, grads)

            players = np.clip(players, 1e-8, 1 - 1e-8)
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
