from typing import Callable

import numpy as np
import pyspiel


class GameAPI:
    def __init__(
        self,
        game_name: str,
        decider: Callable[([int], [int]), int] = lambda x: np.random.choice(x),
    ):
        self.game = pyspiel.load_game(game_name)
        self.state = self.game.new_initial_state()
        self.decider = decider

    def reset(self):
        """Reset the game to the initial state."""
        self.state = self.game.new_initial_state()

    def step(self):
        while not self.state.is_terminal():
            legal_actions = self.state.legal_actions()
            if self.state.is_chance_node():
                outcomes_with_probs = self.state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = self.decider([action_list, prob_list])
                if action not in action_list:
                    raise ValueError(f"Action {action} is not a valid chance outcome.")
                self.state.apply_action(action)
            else:
                if action in legal_actions:
                    self.state.apply_action(action)
                else:
                    raise ValueError(f"Action {action} is not legal.")
            return self.state.observation_tensor()
