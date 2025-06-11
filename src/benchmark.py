import numpy as np
import pyspiel

# The policy_lib import is the key
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import lrs_solver
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.policy import Policy

from RQE import RQE


class RQEBenchmark:
    def __init__(self, game_name, solver):
        self.game = pyspiel.load_game(game_name)
        self.solver = solver
        self.R1, self.R2 = self.extract_payoffs()

    def extract_payoffs(self):
        payoffs = game_payoffs_array(self.game)
        R1 = payoffs[0]
        R2 = payoffs[1]
        return R1, R2

    # The custom JointPolicy class is no longer needed and can be removed.

    class DictPolicy(Policy):
        # ... (This class remains unchanged)
        def __init__(self, game, player_policy, players):
            print(f"Creating DictPolicy for players: {players}")
            super().__init__(game, player_ids=[0, 1])
            self._policy = player_policy

        def action_probabilities(self, state, player_id=None):
            state_str = state.history_str()
            if state_str not in self._policy:
                return {a: 1.0 / state.num_actions() for a in state.legal_actions()}
            return self._policy[state_str]

    def wrap_joint(self, pi1, pi2):
        # ... (This method remains unchanged)
        p0 = {"": {a: pi1[a] for a in range(len(pi1))}}
        p1 = {"": {a: pi2[a] for a in range(len(pi2))}}
        pol0 = self.DictPolicy(self.game, p0, players=[0])
        pol1 = self.DictPolicy(self.game, p1, players=[1])
        return [pol0, pol1]

    def evaluate(self, joint_policies, num_sims=500000):
        # ... (This method remains unchanged)
        returns = np.zeros(2)
        for _ in range(num_sims):
            state = self.game.new_initial_state()
            while not state.is_terminal():
                pid = state.current_player()
                probs = joint_policies[pid].action_probabilities(state)
                actions, ps = zip(*probs.items())
                action = np.random.choice(actions, p=ps)
                state.apply_action(action)
            returns += np.array(state.returns())
        return returns / num_sims

    def exploitability(self, pi1, pi2):
        """
        Calculates NashConv (exploitability) for a 2-player normal-form game.

        This is done directly from the payoff matrices (self.R1, self.R2) and the
        policies, bypassing the pyspiel.nash_conv function which is unsuitable
        for simultaneous-move games.
        """
        # Ensure policies are NumPy arrays for matrix operations
        pi1 = np.array(pi1)
        pi2 = np.array(pi2)

        # R1 is the payoff matrix for player 0 (the row player)
        # R2 is the payoff matrix for player 1 (the column player)

        # --- Calculate current expected payoffs for each player ---
        # The probability of each joint action (i, j) is pi1[i] * pi2[j].
        # The expected payoff is the sum over all (i, j) of: prob(i, j) * payoff(i, j)
        # This is equivalent to the matrix operation: pi1.T * R * pi2
        current_ev_p0 = pi1.T @ self.R1 @ pi2
        current_ev_p1 = pi1.T @ self.R2 @ pi2

        # --- Calculate best-response payoffs for each player ---

        # For Player 0 (row player):
        # What is the best payoff P0 can get against P1's policy (pi2)?
        # 1. Calculate the average payoff for each of P0's pure actions against pi2.
        p0_br_payoffs = (
            self.R1 @ pi2
        )  # Result is a vector of payoffs, one for each of P0's actions
        # 2. The best-response value is the maximum of these payoffs.
        p0_br_value = np.max(p0_br_payoffs)

        # For Player 1 (column player):
        # What is the best payoff P1 can get against P0's policy (pi1)?
        # 1. Calculate the average payoff for each of P1's pure actions against pi1.
        p1_br_payoffs = (
            pi1.T @ self.R2
        )  # Result is a vector of payoffs, one for each of P1's actions
        # 2. The best-response value is the maximum of these payoffs.
        p1_br_value = np.max(p1_br_payoffs)

        # The exploitability for each player is how much they could gain by switching.
        exploitability_p0 = p0_br_value - current_ev_p0
        exploitability_p1 = p1_br_value - current_ev_p1

        # NashConv is the sum of individual exploitabilities.
        return exploitability_p0 + exploitability_p1

    def run(self):
        # ... (This method remains unchanged)
        pi1, pi2 = self.solver.compute_rqe(self.R1, self.R2)
        print("\n--- RQE Policies ---")
        print(f"Player 0: {pi1}")
        print(f"Player 1: {pi2}")

        joint_policy = self.wrap_joint(pi1, pi2)
        avg_returns = self.evaluate(joint_policy)
        nash_conv = self.exploitability(pi1, pi2)

        print("\n--- Evaluation ---")
        print(f"Average returns: {avg_returns}")
        print(f"NashConv (exploitability): {nash_conv:.6f}")

        dev1 = self.policy_deviation(pi1, true_pi1)
        dev2 = self.policy_deviation(pi2, true_pi2)
        print(f"Deviation from True P0 Policy: {dev1:.6f}")
        print(f"Deviation from True P1 Policy: {dev2:.6f}")

    def compute_nash_equilibrium(self):
        """
        Computes one Nash Equilibrium for the game using the Lemke-Howson algorithm.
        """
        # lrs_solver takes the payoff matrices and returns one equilibrium.
        nash_pi1, nash_pi2 = lrs_solver.lemke_howson_2p(self.R1, self.R2)
        return nash_pi1, nash_pi2

    def policy_deviation(self, pi, true_pi):
        """
        Computes the deviation of a policy from the true Nash equilibrium policy.
        """
        pi = np.array(pi)
        true_pi = np.array(true_pi)
        return np.linalg.norm(np.array(pi) - np.array(true_pi))


if __name__ == "__main__":
    game_name = "matrix_rps"
    rqe_solver = RQE(tau1=0.01, tau2=0.02, epsilon1=50, epsilon2=100)
    benchmark = RQEBenchmark(game_name, rqe_solver)

    true_pi1, true_pi2 = benchmark.compute_nash_equilibrium()
    benchmark.run()
