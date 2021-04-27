import logging
import pyspiel
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

from absl import app
from absl import flags
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner, eva, dqn
from open_spiel.python.egt import dynamics
from open_spiel.python.egt import utils
from open_spiel.python.egt import visualization
from open_spiel.python.egt import heuristic_payoff_table
from lbqlearner import LBQLearner


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(5e4),
                     "Number of training episodes.")
flags.DEFINE_integer("num_eval_episodes", int(1e3),
                     "Number of episodes to use during each evaluation.")
flags.DEFINE_integer("eval_freq", int(1e3),
                     "The frequency (in episodes) to run evaluation.")
flags.DEFINE_string("game", "matrix_sh", "Game to load.")  # matrix_rps, matrix_mp, matrix_sh, matrix_cd


def eval_agents(env, agents, num_episodes):
    """
    Evaluate the agents, returning a numpy array of average returns.
    """
    rewards = np.array([0] * env.num_players, dtype=np.float64)
    actions = dict()

    for _ in range(num_episodes):
        time_step = env.reset()

        while not time_step.last():
            agent_output_actions = [
                agents[idx].step(time_step, is_evaluation=True).action
                for idx in range(len(agents))
            ]
            time_step = env.step(agent_output_actions)

            if actions.get(str(agent_output_actions)):
                actions[str(agent_output_actions)] += 1
            else:
                actions[str(agent_output_actions)] = 1

        for i in range(env.num_players):
            rewards[i] += time_step.rewards[i]

    logging.info("Actions: %s", actions)
    return rewards / num_episodes


def train_agents(env, agents):
    """
    Train the agents.
    """
    time_step = env.reset()

    while not time_step.last():
        agent_output_actions = [
            agents[idx].step(time_step).action
            for idx in range(len(agents))
        ]
        time_step = env.step(agent_output_actions)

    for agent in agents:
        agent.step(time_step)

    return


def main(_):
    env = rl_environment.Environment(FLAGS.game)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    game = pyspiel.load_matrix_game(FLAGS.game)
    payoff_tables = utils.game_payoffs_array(game)
    print(payoff_tables)

    agents = [
        # tabular_qlearner.QLearner(player_id=0, num_actions=num_actions),
        LBQLearner(player_id=0, num_actions=num_actions, temperature=0.3, kappa=25),
        LBQLearner(player_id=1, num_actions=num_actions, temperature=0.3, kappa=25)
    ]

    # Train the set of agents
    logging.info("Agent 1: Q-learning | Agent 2: LBQ-learning")
    for cur_episode in range(FLAGS.num_train_episodes+1):
        if cur_episode % int(FLAGS.eval_freq) == 0:
            logging.info("Starting episode %s", cur_episode)
            win_rates = eval_agents(env, agents, FLAGS.num_eval_episodes)
            logging.info("avg_reward Agent 1 vs Agent 2 %s", win_rates)

        train_agents(env, agents)

    return


if __name__ == "__main__":
    app.run(main)
