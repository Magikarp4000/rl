import random
import numpy as np
import matplotlib.pyplot as pl


class Agent:
    def __init__(self, eps, alpha, K, randsize, steps):
        self.eps = eps
        self.alpha = alpha
        self.K = K
        self.randsize = randsize
        self.values = np.array([0 for _ in range(K)], dtype=float)
        self.nums = np.array([0 for _ in range(K)], dtype=float)
        # self.rewards = np.array([random.normalvariate(0, 1) for _ in range(K)])
        self.results = []
        self.totala = np.array([0 for _ in range(steps)], dtype=float)
        self.totalb = np.array([0 for _ in range(steps)], dtype=float)
        self.maxa = 0
        self.maxb = 0
        self.totalmaxa = [0 for _ in range(steps)]
        self.totalmaxb = [0 for _ in range(steps)]
    
    def train(self, steps, agent, init_val=0, step_type='alpha', output_flag=True, recording=False):
        self.maxa = 0
        self.maxb = 0
        self.results.clear()
        self.nums = np.array([0 for _ in range(self.K)], dtype=float)
        self.rewards = np.array([random.normalvariate(0, 1) for _ in range(self.K)])
        self.values = np.array([init_val for _ in range(self.K)], dtype=float)
        record = []
        step = 0
        totalmax = 0
        while step < steps:
            if random.random() < self.eps:
                a = random.randint(0, self.K-1)
            else:
                a = np.argmax(self.values)
            totalmax += max(self.rewards)
            reward = random.normalvariate(self.rewards[a], 1)

            if step_type == 'alpha':
                stepsize = self.alpha
            else:
                stepsize = 0.0 if self.nums[a] == 0 else 1 / self.nums[a]
                
            if agent == 0:
                self.totalmaxa[step] += self.maxa / (step + 1)
                self.maxa += 1 if np.argmax(self.values) == np.argmax(self.rewards) else 0
            else:
                self.totalmaxb[step] += self.maxb / (step + 1)
                self.maxb += 1 if np.argmax(self.values) == np.argmax(self.rewards) else 0
            
            # Learning algorithm
            self.values[a] = self.values[a] + stepsize * (reward - self.values[a])
            # ------------------

            if recording:
                record.append[self.rewards]
            
            self.nums[a] += 1
            for i in range(self.K):
                self.rewards[i] += random.normalvariate(0, self.randsize)
            self.results.append(reward)
            step += 1
        
        # print(max(self.rewards))
        # print(totalmax / steps)
        if output_flag:
            self.output(step_type)
    
    def output(self, step_type='alpha'):
        steps = len(self.results)
        y = np.array([0 for _ in range(steps)], dtype=float)
        total = 0
        for num, result in enumerate(self.results):
            total += result
            y[num] = float(total / (num+1))
        for i in range(steps):
            if step_type == 'alpha':
                self.totala[i] += y[i]
            else:
                self.totalb[i] += y[i]
        

n = 1000
STEPS = 1000
bob = Agent(0.1, 0.1, 10, 0.1, STEPS)

for i in range(n):
    bob.train(STEPS, 0, init_val=5, step_type='alpha', output_flag=True)
    bob.train(STEPS, 1, init_val=0, step_type='alpha', output_flag=True)

x = np.array([i for i in range(1, STEPS+1)], dtype=float)
for i in range(STEPS):
    bob.totala[i] /= n
    bob.totalb[i] /= n
    bob.totalmaxa[i] /= n
    bob.totalmaxb[i] /= n
MARKER_SIZE = 0.1
# pl.scatter(x, bob.totalmaxa, s=MARKER_SIZE)
# pl.scatter(x, bob.totalmaxb, s=MARKER_SIZE)
pl.plot(x, bob.totalmaxa)
pl.plot(x, bob.totalmaxb)
pl.legend(['Agent 0', 'Agent 1'])
pl.title('Correct action %')
pl.show()
