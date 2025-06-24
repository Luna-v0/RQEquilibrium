# File for defining the RQE solution concept
from dataclasses import dataclass
from typing import Callable, Union

from autograd import grad
from autograd import numpy as np

from .opt import (
    ProjectedGradientDescent,
    kl_divergence,
    kl_reversed,
    log_barrier,
    negative_entropy,
    project_simplex,
)


@dataclass
class Player:
    tau: float
    epsilon: float
    game_matrix: np.ndarray


class RQE:
    quantal_function: Callable
    risk_function: Callable

    def __init__(
        self,
        players: list[Player],
        lr=0.1,
        max_iter=500,
        br_iters=50,
        quantal_function: Union[Callable, str] = "log_barrier",
        risk_function: Union[Callable, str] = "kl_divergence",
        projection: Callable = project_simplex,
    ):
        self.players = players
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

    def risk_term(
        self, game: np.ndarray, p: np.ndarray, x: np.ndarray, y: np.ndarray, tau: float
    ) -> float:
        return game.T @ p + (1 / tau) * self.grad_risk(x, y)

    def quantal_term(
        self, game: np.ndarray, p: np.ndarray, x: np.ndarray, epsilon: float
    ) -> float:
        return -game @ x + epsilon * self.grad_quantal(p)

    def optimize(self) -> np.ndarray:
        """
        Optimize the policies for both players using projected gradient descent.
        """

        players = np.random.rand(4, 2)
        players /= np.sum(players, axis=1, keepdims=True)

        pgd = ProjectedGradientDescent(
            lr=self.lr,
            projection=self.projection,
        )

        for _ in range(self.max_iter):
            grads = np.zeros_like(players)

            x = self.quantal_term(
                self.players[0].game_matrix,
                players[0],
                players[1],
                self.players[0].epsilon,
            )
            px = self.risk_term(
                self.players[0].game_matrix,
                players[0],
                players[1],
                players[2],
                self.players[0].tau,
            )
            y = self.quantal_term(
                self.players[1].game_matrix.T,
                players[2],
                players[3],
                self.players[1].epsilon,
            )
            py = self.risk_term(
                self.players[1].game_matrix.T,
                players[2],
                players[3],
                players[0],
                self.players[1].tau,
            )

            grads[0] = x
            grads[1] = px
            grads[2] = y
            grads[3] = py

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
    players = [
        Player(tau=0.005, epsilon=200, game_matrix=R1),
        Player(tau=0.005, epsilon=200, game_matrix=R2),
    ]
    rqe_solver = RQE(players=players, lr=0.01, max_iter=1000, br_iters=50)
    x = rqe_solver.optimize()
    print(x)

    raise ""
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
