# File for defining the RQE solution concept
from dataclasses import dataclass
from typing import Callable, Union

from autograd import grad
from autograd import numpy as np

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

    def risk_term(
        self, game: np.ndarray, x: np.ndarray, p: np.ndarray, y: np.ndarray, tau: float
    ) -> np.array:
        """
        Compute the risk term for a player given the game matrix, policy, and other player's policy.
        """
        return game.T @ x + (1 / tau) * self.grad_risk(p, y)

    def quantal_term(
        self, game: np.ndarray, x: np.ndarray, p: np.ndarray, epsilon: float
    ) -> np.array:
        """
        Compute the quantal response term for a player given the game matrix, policy and epsilon parameter.
        """
        return -game @ p + epsilon * self.grad_quantal(x)

    def optimize(self) -> np.ndarray:
        """
        Optimize the policies for both players using projected gradient descent.
        """

        num_players = len(self.players)  # Number of players
        max_action_set = max(player.game_matrix.shape[1] for player in self.players)

        # Initialize the Projected Gradient Descent optimizer
        pgd = ProjectedGradientDescent(
            lr=self.lr,
            projection=self.projection,
        )

        # Initialize random policies for both players
        policies = np.random.rand(num_players, max_action_set)
        risk_policies = np.random.rand(num_players, max_action_set)
        policies /= np.sum(policies, axis=1, keepdims=True)
        risk_policies /= np.sum(risk_policies, axis=1, keepdims=True)

        print("Pol Shape:", policies.shape)

        for _ in range(self.max_iter):
            # Compute the quantal and risk terms for both players
            policies_buff = policies.copy()
            risk_buff = risk_policies.copy()
            for i, player in enumerate(self.players):
                game = player.game_matrix if i % 2 == 0 else player.game_matrix.T

                quantal_grad = self.quantal_term(
                    game, policies_buff[i], risk_buff[i], player.epsilon
                )
                opponnet_policies = np.delete(policies_buff, i, axis=0)

                risk_grad = self.risk_term(
                    game,
                    policies_buff[i],
                    risk_buff[i],
                    opponnet_policies,
                    player.tau,
                )

                policies[i] = pgd.step(policies_buff[i], quantal_grad)
                risk_policies[i] = pgd.step(risk_buff[i], risk_grad)

        return policies

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
    R3 = R1.copy()
    R4 = R2.copy()
    RQE.print_game(R1, R2)
    players = [
        Player(tau=0.001, epsilon=170, game_matrix=R1),
        Player(tau=0.06, epsilon=110, game_matrix=R2),
        Player(tau=0.003, epsilon=190, game_matrix=R3),
        Player(tau=0.05, epsilon=130, game_matrix=R4),
    ]
    rqe_solver = RQE(players=players, lr=1e-4, max_iter=1000, br_iters=50)

    print("Computed policies for RQE")
    # print("Player 1:", np.argmax(pi1), "with policy:", pi1)
    i = 0
    for pi in rqe_solver.optimize():
        print(f"Player {i + 1}: Best Action {np.argmax(pi)} with policy: {pi}")
        i += 1
