#!/usr/bin/env python3
# encoding: utf-8
"""
fcpa_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2021 KU Leuven. All rights reserved.
"""

import sys
import argparse
import os
import logging
import numpy as np
import pyspiel
import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.algorithms import deep_cfr_tf2


logger = logging.getLogger('be.kuleuven.cs.dtai.fcpa')


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play FCPA poker.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.
        """
        pyspiel.Bot.__init__(self)
        fcpa_game_string = pyspiel.hunl_game_string("fcpa")
        game = pyspiel.load_game(fcpa_game_string)
        tf.enable_eager_execution()
        self.player_id = player_id
        self.deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
            game,
            policy_network_layers=(64, 128, 128, 64),
            advantage_network_layers=(64, 64, 64, 64))
        model_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_models/model140")
        self.deep_cfr_solver._policy_network = tf.keras.models.load_model(model_loc)

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        pass

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        pass

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        """
        prob_values = self.action_probabilities(state).values()
        prob_list = [val / sum(prob_values) for val in prob_values]
        rng = np.random.RandomState()
        legal_actions = state.legal_actions(self.player_id)
        action = rng.choice(legal_actions, p=prob_list)
        return action

    def action_probabilities(self, state):
        """Returns action probabilities dict for a single batch."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        legal_actions_mask = tf.constant(
            state.legal_actions_mask(cur_player), dtype=tf.float32)
        info_state_vector = tf.constant(
            state.information_state_tensor(), dtype=tf.float32)
        if len(info_state_vector.shape) == 1:
            info_state_vector = tf.expand_dims(info_state_vector, axis=0)
        if len(legal_actions_mask.shape) == 1:
            legal_actions_mask = tf.expand_dims(legal_actions_mask, axis=0)
        probs = self.deep_cfr_solver._policy_network((info_state_vector, legal_actions_mask), training=False)
        probs = probs.numpy()
        return {action: probs[0][action] for action in legal_actions}


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0, 1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())
