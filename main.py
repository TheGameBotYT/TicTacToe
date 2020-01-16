import numpy as np
import pandas as pd
from env import TicTacToeEnv
from agent import QLearningAgent

env = TicTacToeEnv()
agent = QLearningAgent(env)

for game_nr in range(1000000):
    if game_nr % 10000 == 0:
        print(game_nr)
    done = False
    s = env.reset().copy()
    # print('Init', s)
    while not done:
        a = agent.take_action(s)
        r, s_, done, _ = env.step(a)
        agent.learn(s, a, r, s_, done)
        # print(s, a, r, s_, done)
        s = s_.copy()

V = pd.DataFrame.from_dict(agent._V, orient='index', dtype=np.float32, columns=['V'])
N = pd.DataFrame.from_dict(agent._N, orient='index', dtype=np.uint32, columns=['N'])
df = V.merge(N, how='left', left_index=True, right_index=True)
states = pd.DataFrame(df.index.values.tolist(), index=df.index)
res = states.merge(V, how='left',
                   left_index=True, right_index=True).merge(N, how='left',
                                                            left_index=True, right_index=True).reset_index(drop=True)
res.to_pickle('test.p')