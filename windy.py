from agents import *
from racetrack_utils import get_track


class Windy(TDAgent):
    def __init__(self, grid, start, finish, wind):
        super().__init__()
        self.config(['grid', 'start', 'finish', 'wind'])
        self.grid = grid
        self.start = start
        self.finish = finish
        self.wind = wind
        self.W, self.H = np.array(self.grid).max(axis=0) + 1
        self.mW, self.mH = np.array(self.grid).min(axis=0) + 1
        self.core_init()
    
    def set_state_actions(self):
        self.states = self.grid
        print(self.states)
        self.actions = [[(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)]
                        for _ in self.states]
        for s in range(len(self.states)):
            self.actions[s].remove((0, 0))
        self.actions.append([(0, 0)])

    def next_state(self, s, a):
        x, y = self.states[s]
        dx, dy = self.actions[s][a]
        tempx = np.clip(x + dx, self.mW, self.W - 1)
        tempy = np.clip(y + dy, self.mW, self.W - 1)
        nx = tempx
        ny = tempy + self.wind[tempy][tempx] + random.randint(-1, 1)
        nx = np.clip(nx, self.mW, self.W - 1)
        ny = np.clip(ny, self.mH, self.H - 1)
        reward = -1
        if (nx, ny) in self.finish:
            return self.size, reward
        return self.state_to_index((nx, ny)), reward


grid, start, finish = get_track()
wind = [[0 for i in range(100)] for j in range(100)]
agent = Windy(grid, start, finish, wind)
agent.train('onpolicy', 10000, batch_size=10)
print(agent.q)
