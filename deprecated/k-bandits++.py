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
    
    def train(self, steps, step_type='alpha', output_flag=True):
        self.maxa = 0
        self.maxb = 0
        self.results.clear()
        self.nums = np.array([0 for _ in range(self.K)], dtype=float)
        self.rewards = np.array([random.normalvariate(0, 1) for _ in range(self.K)])
        self.values = np.array([0 for _ in range(self.K)], dtype=float)
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
                self.maxa += 1 if np.argmax(self.values) == np.argmax(self.rewards) else 0
            else:
                stepsize = 0.0 if self.nums[a] == 0 else 1 / self.nums[a]
                self.maxb += 1 if np.argmax(self.values) == np.argmax(self.rewards) else 0
            
            # Learning algorithm
            self.values[a] = self.values[a] + stepsize * (reward - self.values[a])
            # ------------------

            self.nums[a] += 1
            for i in range(self.K):
                self.rewards[i] += random.normalvariate(0, self.randsize)
            self.results.append(reward)
            step += 1
        
        print(max(self.rewards))
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
        

n = 200
steps = 5000
bob = Agent(0.1, 0.1, 10, 0.10, steps)

cura = 0.0
curb = 0.0
for i in range(n):
    bob.train(steps, step_type='alpha', output_flag=True)
    cura += bob.maxa / steps
    bob.train(steps, step_type='SA', output_flag=True)
    curb += bob.maxb / steps
print(cura/n)
print(curb/n)
x = np.array([i for i in range(1, steps+1)], dtype=float)
for i in range(steps):
    bob.totala[i] /= n
    bob.totalb[i] /= n
pl.plot(x, bob.totala)
pl.plot(x, bob.totalb)
pl.legend(['alpha', 'SA'])
pl.show()
