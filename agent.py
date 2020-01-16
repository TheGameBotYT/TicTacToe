from collections import defaultdict
import numpy as np

class QLearningAgent(object):
    def __init__(self, env, epsilon=0.2, alpha=0.1, gamma=1):
        self._V = defaultdict(lambda: 0)
        self._N = defaultdict(lambda: 0)
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def learn(self, s, a, r, s_, done):
        self._V[tuple(s)] = (1-self.alpha) * self.get_state_value(s) + self.alpha * (r + self.gamma * self.get_state_value(s_))
        self._N[tuple(s)] += 1

    def get_state_value(self, state):
        val = self._V[tuple(state)]
        return val

    def take_action(self, state):
        viable_actions = self.env.get_viable_actions(state)

        if np.random.uniform() < self.epsilon:
            return np.random.choice(viable_actions)
        else:
            return self.get_best_action(state, viable_actions)

    def get_best_action(self, state, viable_actions):
        next_states = [self.env.draw_next_state(a) for a in viable_actions]
        next_state_vals = [self._V[tuple(s)] for s in next_states]
        if state[0] == 1:
            max_V = np.max(next_state_vals)
            return np.random.choice([a for a, v in zip(viable_actions, next_state_vals) if v == max_V])
        elif state[0] == 2:
            min_V = np.min(next_state_vals)
            return np.random.choice([a for a, v in zip(viable_actions, next_state_vals) if v == min_V])
        else:
            raise NotImplementedError