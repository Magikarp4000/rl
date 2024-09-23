from agents import *
import racetrack_utils


class Maze(Dyna):
    def __init__(self, cells, starts, finish):
        super().__init__()
        self.config(['cells', 'starts', 'finish'])
        self.cells = cells
        self.starts = starts
        self.finish = finish
        self.core_init()
    
    def load_convert(self):
        super().load_convert()
        self.cells = [tuple(x) for x in self.cells]
        self.starts = [tuple(x) for x in self.starts]
        self.finish = [tuple(x) for x in self.finish]

    def set_state_actions(self):
        self.cells_set = set(self.cells)
        self.starts_set = set(self.starts)
        self.finish_set = set(self.finish)
        self.states = [(x, y) for x, y in self.cells]
        self.actions = [[(1, 0), (-1, 0), (0, 1), (0, -1)] for _ in self.states]
        self.actions.append([(0, 0)])

    def next_state(self, s, a):
        x, y = self.states[s]
        dx, dy = self.actions[s][a]
        nx = x + dx
        ny = y + dy
        reward = -1
        if (nx, ny) in self.finish_set:
            return self.size, reward
        if (nx, ny) not in self.cells_set:
            return s, reward
        new_s = self.state_to_index((nx, ny))
        return new_s, reward
    
    def test(self, num_tests):
        total_steps = 0
        for _ in range(num_tests):
            s, a = self.init_ep()
            while s < self.size:
                a = self.pi[s]
                s, reward = self.next_state(s, a)
                total_steps += 1
        avg_steps = total_steps / num_tests
        print(f"Avg num of steps: {avg_steps}")


cells, starts, finish = racetrack_utils.get_track()
pro = Maze(cells, starts, finish)
pro.load('maze/v0.2-alpha', load_actions=True)
pro.train('q+', 10000, n=10, kappa=0.05, batch_size=1000)
# pro.save('maze/v0.2-alpha')
pro.test(10000)