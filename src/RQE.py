# File for defining the RQE solution concept
from dataclasses import dataclass
from typing import Callable, Union

import matplotlib

matplotlib.use(
    "module://matplotlib-backend-kitty"
)  # Use TkAgg backend for interactive plots


from autograd import grad
from autograd import numpy as np
from matplotlib import pyplot as plt

from opt import (
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

    def quantal_term(
        self, game: np.ndarray, p: np.ndarray, x: np.ndarray, epsilon: float
    ) -> float:
        """
        Compute the quantal term for a given game matrix, policy, and epsilon.
        """
        return -game @ p + epsilon * self.grad_quantal(x)

    def risk_term(
        self, game: np.ndarray, p: np.ndarray, x: np.ndarray, y: np.ndarray, tau: float
    ) -> float:
        """
        Compute the risk term for a given game matrix, policy, and epsilon.
        """
        return game.T @ x + (1 / tau) * self.grad_risk(p, y)

    def utility(
        self, game: np.ndarray, p: np.ndarray, x: np.ndarray, y: np.ndarray, tau: float
    ) -> float:
        """
        Compute the utility for a player given their policy and the opponent's policy.
        """
        return -x.T.dot(game).dot(p) - (1 / tau) * self.risk_function(y, p)

    def optimize(self) -> np.ndarray:
        """
        Optimize the policies for both players using projected gradient descent.
        """

        (n, m) = self.players[0].game_matrix.shape

        x = self.projection(np.ones([n, 1]))
        y = self.projection(np.ones([m, 1]))
        px = self.projection(np.ones([n, 1]))
        py = self.projection(np.ones([m, 1]))

        pgd = ProjectedGradientDescent(
            lr=self.lr,
            projection=self.projection,
        )

        payoffs = []
        lasts = [0 for _ in range(len(self.players))]
        lasti = 100

        for t in range(self.max_iter):

            quantal_x = self.quantal_term(
                self.players[0].game_matrix, px, x, self.players[0].epsilon
            )
            risk_x = self.risk_term(
                self.players[0].game_matrix, px, x, y, self.players[0].tau
            )
            quantal_y = self.quantal_term(
                self.players[1].game_matrix, py, y, self.players[1].epsilon
            )
            risk_y = self.risk_term(
                self.players[1].game_matrix, py, y, x, self.players[1].tau
            )

            x = pgd.step(x, quantal_x)
            px = pgd.step(px, risk_x)
            y = pgd.step(y, quantal_y)
            py = pgd.step(py, risk_y)

            print(x)
            break
            arr = [x, y]

            utility_1 = self.utility(
                self.players[0].game_matrix, px, x, y, self.players[0].tau
            )
            utility_2 = self.utility(
                self.players[1].game_matrix, py, y, x, self.players[1].tau
            )

            payoffs.append((utility_1, utility_2))
            diffs = [arr[i] - lasts[i] for i in range(len(arr))]
            lasts = [arr[i] for i in range(len(arr))]

        payoffs_values = [np.mean(payoff[-lasti:]) for payoff in payoffs]

        # plt.plot([p[0] for p in payoffs], label="Player 1 Payoff")
        # plt.pause(0.1)
        return x, y, x, y
        return payoffs_values[0], payoffs_values[1], x, y

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
    R1 = np.array([[200, 160], [370, 10]])
    R2 = np.array([[160, 10], [200, 370]])
    RQE.print_game(R1, R2)
    players = [
        Player(tau=0.001, epsilon=170, game_matrix=R1),
        Player(tau=0.06, epsilon=110, game_matrix=R2),
    ]
    rqe_solver = RQE(players=players, lr=0.01, max_iter=1000, br_iters=50)
    p1, p2, pi1, pi2 = rqe_solver.optimize()
    print("Computed policies for RQE:")
    print(f"Player 1: Best Action {np.argmax(pi1)} with policy: {pi1}")
    print(f"Player 2: Best Action {np.argmax(pi2)} with policy: {pi2}")

    print("Payoffs:")
    print(f"Player 1: {p1}, Player 2: {p2}")

    print(f"Expected pi_1 pi_2 strategies: {np.array([[0.47, 0.53], [0.65, 0.35]])}")

    import nashpy as nash

    game = nash.Game(R1, R2)
    equilibrium = game.support_enumeration()
    print("Nash Equilibrium using nashpy:")
    for eq in equilibrium:
        print(f"Player 1: Best Action {np.argmax(eq[0])} with policy: {eq[0]}")
        print(f"Player 2: Best Action {np.argmax(eq[1])} with policy: {eq[1]}")

# print("Computed policies for RQE")
# # print("Player 1:", np.argmax(pi1), "with policy:", pi1)
# print(f"Player 1: Best Action {np.argmax(pi1)} with policy: {pi1}")
# # print("Player 2:", np.argmax(pi2), "with policy:", pi2)
# print(f"Player 2: Best Action {np.argmax(pi2)} with policy: {pi2}")

# import nashpy as nash

# game = nash.Game(R1, R2)
# equilibrium = game.support_enumeration()
# print("Nash Equilibrium using nashpy:")

# for eq in equilibrium:
#     print(f"Player 1: Best Action {np.argmax(eq[0])} with policy: {eq[0]}")
#     print(f"Player 2: Best Action {np.argmax(eq[1])} with policy: {eq[1]}")
