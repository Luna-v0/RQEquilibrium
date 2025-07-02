import random

import numpy as np
import open_spiel.python as oas
import pyspiel
from open_spiel.python import games

game = pyspiel.load_matrix_game("blotto(players=3)")
state = game.new_initial_state()
help(game)


# while not state.is_terminal():
legal_actions = state.legal_actions()
# I need to know which player is taking the action.
print(game.max_game_length())
print("state", legal_actions)
if state.is_simultaneous_node():
    print("Simultaneous node detected.")

else:
    print("Sequential node detected.")

    # if state.is_chance_node():
    #    # Sample a chance event outcome.
    #    outcomes_with_probs = state.chance_outcomes()
    #    action_list, prob_list = zip(*outcomes_with_probs)
    #    print(f"Action list: {action_list}, Probabilities: {prob_list}")
    #    action = np.random.choice(action_list, p=prob_list)
    #    state.apply_action(action)
    # else:
    #    # The algorithm can pick an action based on an observation (fully observable
    #    # games) or an information state (information available for that player)
    #    # We arbitrarily select the first available action as an example.
    #    action = legal_actions[0]
    #    state.apply_action(action)
