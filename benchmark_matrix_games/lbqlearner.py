import numpy as np
import math
import collections

from open_spiel.python import rl_agent
from open_spiel.python.algorithms.tabular_qlearner import QLearner


class LBQLearner(QLearner):
    """
    Lenient Boltzmann Q-learner for matrix games
    """

    def __init__(self, player_id, num_actions, temperature=1., kappa=10, discount_factor=1.):
        super().__init__(player_id, num_actions, discount_factor)
        self._temperature = temperature
        self._prev_rewards = dict()
        self._kappa = kappa

        for i in range(num_actions):
            self._prev_rewards[i] = np.array([])

    def _boltzmann_distribution(self, info_state, legal_actions, temperature):
        probs = np.zeros(self._num_actions)
        sum_values = 0.0

        for i in range(self._num_actions):
            temp = math.exp(self._q_values[info_state][i]/temperature)
            probs[i] = temp
            sum_values += temp

        probs /= sum_values
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs

    def step(self, time_step, is_evaluation=False):
        if self._centralized:
            info_state = str(time_step.observations["info_state"])
        else:
            info_state = str(time_step.observations["info_state"][self._player_id])
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        # Prevent undefined errors if this agent never plays until terminal step
        action, probs = None, None

        # Act step: don't act at terminal states.
        if not time_step.last():
            action, probs = self._boltzmann_distribution(info_state, legal_actions, temperature=self._temperature)

        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            target = time_step.rewards[self._player_id]
            self._prev_rewards[self._prev_action] = np.append(self._prev_rewards[self._prev_action], target)

            # only update q value when we have kappa number of rewards to look at
            for action_key in self._prev_rewards:
                if self._prev_rewards[action_key].size >= self._kappa:
                    prev_q_value = self._q_values[self._prev_info_state][action_key]
                    target = self._prev_rewards[action_key].max(initial=-np.inf)
                    self._last_loss_value = target - prev_q_value
                    self._q_values[self._prev_info_state][action_key] += (
                            self._step_size * self._last_loss_value)
                    self._prev_rewards[action_key] = np.array([])

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)
