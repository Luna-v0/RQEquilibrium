from typing import Callable

import numpy as np
import pyspiel


def get_matrix(game_name: str) -> list[np.ndarray]:
    """
    Load a matrix game from OpenSpiel and return its payoff matrix.

    Args:
        game_name (str): The name of the game to load.

    Returns:
        np.ndarray: The payoff matrix of the game.
    """
    print(game_name)
    game = pyspiel.load_matrix_game(game_name)
    return [game.player_utilities(player) for player in range(game.num_players())]
