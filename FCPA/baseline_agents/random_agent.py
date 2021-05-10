import sys
import argparse
import logging
import numpy as np
import pyspiel
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.bots import uniform_random


def get_agent_for_tournament(player_id, seed=None):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    policy = pyspiel.PreferredActionPolicy([1, 0])

    if seed is None:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(seed)

    return uniform_random.UniformRandomBot(player_id, rng)
