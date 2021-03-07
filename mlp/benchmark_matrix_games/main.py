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

flags.DEFINE_integer("num_train_episodes", int(3e4),
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


def draw_replicator_dynamics_3x3():
    game = pyspiel.load_game(FLAGS.game)
    payoff_tensor = utils.game_payoffs_array(game)
    dyn = dynamics.SinglePopulationDynamics(payoff_tensor, dynamics.replicator)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3x3")
    ax.quiver(dyn)
    ax.set(labels=["Rock", "Paper", "Scissors"])
    plt.savefig("images/directional_field_" + FLAGS.game + ".png")

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3x3")
    ax.streamplot(dyn)
    ax.set(labels=["Rock", "Paper", "Scissors"])
    plt.savefig("images/streamline_" + FLAGS.game + ".png")

    return


def draw_replicator_dynamics_2x2():
    game = pyspiel.load_game(FLAGS.game)
    payoff_tensor = utils.game_payoffs_array(game)
    dyn = dynamics.MultiPopulationDynamics(payoff_tensor, dynamics.replicator)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="2x2")
    ax.quiver(dyn)

    if FLAGS.game == "matrix_mp":
        ax.set_title("Matching Pennies")
        ax.set_xlabel("Pr(Heads)")
        ax.set_ylabel("Pr(Heads)")
    elif FLAGS.game == "matrix_sh":
        ax.set_title("Stag Hunt")
        ax.set_xlabel("Pr(Stag)")
        ax.set_ylabel("Pr(Stag)")
    elif FLAGS.game == "matrix_cd":
        ax.set_title("Chicken-Dare")
        ax.set_xlabel("Pr(Dare)")
        ax.set_ylabel("Pr(Dare)")

    plt.savefig("images/directional_field_" + FLAGS.game + ".png")

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="2x2")
    ax.streamplot(dyn)

    if FLAGS.game == "matrix_mp":
        ax.set_title("Matching Pennies")
        ax.set_xlabel("Pr(Heads)")
        ax.set_ylabel("Pr(Heads)")
    elif FLAGS.game == "matrix_sh":
        ax.set_title("Stag Hunt")
        ax.set_xlabel("Pr(Stag)")
        ax.set_ylabel("Pr(Stag)")
    elif FLAGS.game == "matrix_cd":
        ax.set_title("Chicken-Dare")
        ax.set_xlabel("Pr(Dare)")
        ax.set_ylabel("Pr(Dare)")

    plt.savefig("images/streamline_" + FLAGS.game + ".png")

    return


def main(_):
    env = rl_environment.Environment(FLAGS.game)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    game = pyspiel.load_matrix_game(FLAGS.game)
    payoff_tables = utils.game_payoffs_array(game)
    print(payoff_tables)

    state_size = env.observation_spec()["info_state"][0]
    sess1 = tf.Session()  # not sure if multiple sessions are needed
    sess2 = tf.Session()
    sess3 = tf.Session()

    agents1 = [
        tabular_qlearner.QLearner(player_id=0, num_actions=num_actions),
        # LBQLearner(player_id=1, num_actions=num_actions, temperature=1)
        tabular_qlearner.QLearner(player_id=1, num_actions=num_actions)
        # eva.EVAAgent(session=sess, player_id=1, state_size=state_size, num_actions=num_actions, game=env)
    ]

    agents2 = [
        tabular_qlearner.QLearner(player_id=0, num_actions=num_actions),
        dqn.DQN(player_id=1, num_actions=num_actions, state_representation_size=state_size, session=sess1)
    ]

    agents3 = [
        dqn.DQN(player_id=0, num_actions=num_actions, state_representation_size=state_size, session=sess2),
        dqn.DQN(player_id=1, num_actions=num_actions, state_representation_size=state_size, session=sess3)
    ]

    sess1.run(tf.global_variables_initializer())
    sess2.run(tf.global_variables_initializer())
    sess3.run(tf.global_variables_initializer())

    if FLAGS.game == "matrix_rps":
        draw_replicator_dynamics_3x3()
    else:
        draw_replicator_dynamics_2x2()

    # Train the first set of agents
    logging.info("Agent 1: Q-learning | Agent 2: Q-learning")
    for cur_episode in range(FLAGS.num_train_episodes+1):
        if cur_episode % int(FLAGS.eval_freq) == 0:
            logging.info("Starting episode %s", cur_episode)
            win_rates = eval_agents(env, agents1, FLAGS.num_eval_episodes)
            logging.info("avg_reward Agent 1 vs Agent 2 %s", win_rates)

        train_agents(env, agents1)

    # Train the second set of agents
    logging.info("Agent 1: Q-learning | Agent 2: DQN")
    for cur_episode in range(FLAGS.num_train_episodes + 1):
        if cur_episode % int(FLAGS.eval_freq) == 0:
            logging.info("Starting episode %s", cur_episode)
            win_rates = eval_agents(env, agents2, FLAGS.num_eval_episodes)
            logging.info("avg_reward Agent 1 vs Agent 2 %s", win_rates)

        train_agents(env, agents2)

    # Train the third set of agents
    logging.info("Agent 1: DQN | Agent 2: DQN")
    for cur_episode in range(FLAGS.num_train_episodes + 1):
        if cur_episode % int(FLAGS.eval_freq) == 0:
            logging.info("Starting episode %s", cur_episode)
            win_rates = eval_agents(env, agents3, FLAGS.num_eval_episodes)
            logging.info("avg_reward Agent 1 vs Agent 2 %s", win_rates)

        train_agents(env, agents3)

    return


if __name__ == "__main__":
    app.run(main)
