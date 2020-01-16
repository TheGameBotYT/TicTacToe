import numpy as np


class TicTacToeEnv(object):
    def __init__(self):
        """
        self.state is encoded with the current player first and the 9 positions on the board later
        0 -> No mark
        1 -> 'X' mark
        2 -> 'O' mark
        S, A encoding of the grid:
        1 | 2 | 3
        4 | 5 | 6
        7 | 8 | 9
        """
        self.state = np.array([])

    def step(self, action):
        if action not in self.get_viable_actions(self.state):
            return 0, self.state, False, None
        else:
            next_state = self.draw_next_state(action)
            self.update_internal_state(next_state)
            reward, done_flag = self.end_condition(next_state)
        return reward, next_state, done_flag, None

    def get_viable_actions(self, state):
        return [a for a in range(1, 10) if self.state[a] == 0]

    def draw_next_state(self, action):
        next_state = self.state.copy()
        if next_state[0] == 1:
            next_state[action] = 1
            next_state[0] = 2
        elif next_state[0] == 2:
            next_state[action] = 2
            next_state[0] = 1
        else:
            raise NotImplementedError
        return next_state

    def update_internal_state(self, state):
        self.state = state

    def end_condition(self, state):
        # Indexed +1 due to current player state
        for index_group in [[1, 2, 3], [4, 5, 6], [7, 8, 9],
                            [1, 4, 7], [2, 5, 8], [3, 6, 9],
                            [1, 5, 9], [3, 5, 7]]:
            if all(x == 1 for x in state[index_group]):
                return 1, True
            elif all(x == 2 for x in state[index_group]):
                return -1, True
            else:
                pass
        if len(self.get_viable_actions(state)) == 0:
            return 0, True

        return 0, False

    def reset(self):
        self.first_player = np.random.choice([1, 2])
        init_state = np.array([self.first_player,
                               0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.state = init_state
        return init_state

    def render(self, state):
        value_to_str = [self.state_value_to_render(val) for val in state]
        print(value_to_str[1], '|', value_to_str[2], '|', value_to_str[3], '\n', value_to_str[4], '|', value_to_str[5],
              '|', value_to_str[6], '\n', value_to_str[7], '|', value_to_str[8], '|', value_to_str[9])

    def _state_value_to_render(self, state_value):
        if state_value == 0:
            return ' '
        elif state_value == 1:
            return 'x'
        elif state_value == 2:
            return 'o'
        else:
            raise NotImplementedError