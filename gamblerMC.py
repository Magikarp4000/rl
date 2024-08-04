import random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from agents import *


class Gambler(MCAgent):
    def __init__(self, n=0, ph=0.5):
        super().__init__()
        self.configure(['n', 'ph'])
        
        # Set environment variables
        self.n = n
        self.ph = ph

        # Set states and actions
        self.core_init()
    
    def set_state_actions(self):
        self.states = [i for i in range(1, self.n)]
        self.actions = [[action for action in range(1, min(state, self.n - state) + 1)]
                        for state in self.states]
    
    def get_episode(self):
        seq = []
        coins = random.randint(1, self.n - 1)
        while coins > 0 and coins < self.n:
            s = self.state_to_index(coins)
            a = self.get_action(s)
            action = self.actions[s][a]
            if random.random() <= self.ph:
                coins += action
            else:
                coins -= action
            reward = 0
            if coins >= self.n:
                reward = 1
            seq.append((s, a, reward))
        return seq
    
    def test(self):
        coins = random.randint(1, self.n - 1)
        while coins > 0 and coins < self.n:
            s = self.state_to_index(coins)
            action = self.actions[s][self.pi[s]]
            if random.random() <= self.ph:
                coins += action
            else:
                coins -= action
        return self.n if coins >= n else 0
    
    # def play(self):
    #     coins = random.randint(1, self.n - 1)
    #     print()
    #     while coins > 0 and coins < self.n:
    #         print(f"You have {coins} coins")
    #         action = self.get_user_action(coins)
    #         if random.random() <= self.ph:
    #             coins += action
    #             print("\nYou won the bet")
    #         else:
    #             coins -= action
    #             print("\nYou lost the bet")
    #     if coins >= n:
    #         print("You won!!! What a pro!!!")
    #     else:
    #         print("You lost all your coins noob")
    
    # def get_user_action(self, state):
    #     _min = 1
    #     _max = state
    #     action = _max + 1
    #     while action < _min or action > _max:
    #         try:
    #             action = int(input(f"Number of coins to bet ({_min} to {_max})? "))
    #             if action < _min or action > _max:
    #                 print("Invalid action bro")
    #         except ValueError:
    #             print("Invalid action bro")
    #     return action
    
    def animate(self):
        pl.plot([i for i in range(1, self.n)], [self.actions[s][self.pi[s]] for s in range(self.size)])
        pl.show()


n = 100
chad = Gambler(n=n, ph=0.75)
chad.train('onpolicy', 10000, eps=0.1)
print(chad.q)
chad.animate()
num_tests = 10000
total = 0
for i in range(num_tests):
    total += chad.test()
print(total / num_tests)
