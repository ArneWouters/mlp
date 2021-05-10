from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import six

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

from open_spiel.python.algorithms import deep_cfr_tf2
from baseline_agents import check_call_agent, fold_call_agent, fold_agent, random_agent
import pyspiel


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_games", 3000, "Number of games to simulate")
flags.DEFINE_integer("seed", 12761381, "The seed to use for the RNG.")

# Supported types of players: "random", "fold", "check_call", "call_fold"
flags.DEFINE_string("player1", "call_fold", "Type of the agent for player 1.")


def LoadAgent(agent_type, game, player_id, rng):
    """Return a bot based on the agent type."""
    if agent_type == "random":
        return random_agent.get_agent_for_tournament(player_id, FLAGS.seed)
    elif agent_type == "check_call":
        return check_call_agent.get_agent_for_tournament(player_id, FLAGS.seed)
    elif agent_type == "fold":
        return fold_agent.get_agent_for_tournament(player_id, FLAGS.seed)
    elif agent_type == "call_fold":
        return fold_call_agent.get_agent_for_tournament(player_id, FLAGS.seed)
    else:
        raise RuntimeError("Unrecognized agent type: {}".format(agent_type))


def drawGraph(x, y):
    plt.plot(x, y)
    plt.legend(['Utility vs ' + FLAGS.player1])
    plt.show()


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
    probs = self._policy_network((info_state_vector, legal_actions_mask),
                                 training=False)
    probs = probs.numpy()
    return {action: probs[0][action] for action in legal_actions}


def getDeepCFRAgent(game):
    deep_cfr_solver = deep_cfr_tf2.DeepCFRSolver(
        game,
        policy_network_layers=(64, 128, 128, 64),
        advantage_network_layers=(64, 64, 64, 64))
    deep_cfr_solver._policy_network = tf.keras.models.load_model("saved_models/iter_model120")
    return deep_cfr_solver


def load_agents(game, agents=None):
    rng = np.random.RandomState(FLAGS.seed)

    if agents is None:
        agents = [
            getDeepCFRAgent(game),
            LoadAgent(FLAGS.player1, game, 1, rng)
        ]
        return agents

    if isinstance(agents[0], deep_cfr_tf2.DeepCFRSolver):
        return [LoadAgent(FLAGS.player1, game, 0, rng), agents[0]]
    else:
        return [agents[1], LoadAgent(FLAGS.player1, game, 1, rng)]


def main(_):
    tf.enable_eager_execution()
    fcpa_game_string = pyspiel.hunl_game_string("fcpa")
    game = pyspiel.load_game(fcpa_game_string)

    rng = np.random.RandomState(FLAGS.seed)
    agents = load_agents(game)

    utilities = np.array([0] * len(agents), dtype=np.float64)
    utilityHistory = np.array([])
    x = np.array([])

    for i in range(FLAGS.num_games):
        state = game.new_initial_state()

        while not state.is_terminal():
            # The state can be three different types: chance node,
            # simultaneous node, or decision node
            current_player = state.current_player()
            if state.is_chance_node():
                # Chance node: sample an outcome
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = rng.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                # Decision node: sample action for the single current player
                legal_actions = state.legal_actions()
                if isinstance(agents[current_player], deep_cfr_tf2.DeepCFRSolver):
                    prob_values = action_probabilities(agents[current_player], state).values()
                    prob_list = [val/sum(prob_values) for val in prob_values]
                    action = rng.choice(legal_actions, p=prob_list)
                else:
                    action = agents[current_player].step(state)

                state.apply_action(action)

        # Game is now done. Print utilities for each player
        returns = state.returns()
        for pid in range(game.num_players()):
            utilities[pid] += returns[pid]
            if isinstance(agents[pid], deep_cfr_tf2.DeepCFRSolver):
                utilityHistory = np.append(utilityHistory, utilities[pid])

        x = np.append(x, i)
        # print(utilities)
        utilities[0], utilities[1] = utilities[1], utilities[0]
        agents = load_agents(game, agents)

    drawGraph(x, utilityHistory)
    # print(utilities)
    for pid in range(game.num_players()):
        print("Player {}: {} mbb/g".format(pid, utilities[pid]*10/FLAGS.num_games))


if __name__ == "__main__":
    app.run(main)
