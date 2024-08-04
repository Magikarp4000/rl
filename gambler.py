import random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from agents import *


class Gambler(DPAgent):
    def __init__(self, n=0, ph=0.5):
        super().__init__()
        self.config(['n', 'ph'])
        
        # Set environment variables
        self.n = n
        self.ph = ph

        # Environment
        self.p = []

        # Set states and actions
        self.core_init()
    
    def set_state_actions(self):
        self.states = [i for i in range(1, self.n)]
        self.actions = [[action for action in range(1, min(state, self.n - state) + 1)]
                        for state in self.states]
    
    def get_prob(self, s, a):
        return self.p[s][a], 0

    def set_dynamics(self):
        start_time = time.time()
        self.p = [[{} for _ in range(len(self.actions[s]))] for s in range(self.size)]
        for s, state in enumerate(self.states):
            for a, action in enumerate(self.actions[s]):
                win_state = state + action
                lose_state = state - action
                win_s = lose_s = self.size
                win_reward, lose_reward = 1, 0
                if win_state < self.n:
                    win_s = self.state_to_index(win_state)
                    win_reward = 0
                if lose_state > 0:
                    lose_s = self.state_to_index(lose_state)
                add_to_dict(self.p[s][a], (win_s, win_reward), self.ph)
                add_to_dict(self.p[s][a], (lose_s, lose_reward), 1 - self.ph)
        end_time = time.time()
        print(f"Finished setting environment dynamics in {round(end_time - start_time, 2)}s.")
    
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
    
    def play(self):
        coins = random.randint(1, self.n - 1)
        print()
        while coins > 0 and coins < self.n:
            print(f"You have {coins} coins")
            action = self.get_user_action(coins)
            if random.random() <= self.ph:
                coins += action
                print("\nYou won the bet")
            else:
                coins -= action
                print("\nYou lost the bet")
        if coins >= n:
            print("You won!!! What a pro!!!")
        else:
            print("You lost all your coins noob")
    
    def get_user_action(self, state):
        _min = 1
        _max = state
        action = _max + 1
        while action < _min or action > _max:
            try:
                action = int(input(f"Number of coins to bet ({_min} to {_max})? "))
                if action < _min or action > _max:
                    print("Invalid action bro")
            except ValueError:
                print("Invalid action bro")
        return action
    
    def animate(self):
        pl.plot([i for i in range(1, n)], [chad.actions[s][chad.pi[s]] for s in range(chad.size)])
        pl.show()


def add_to_dict(_dict, _key, _value):
    if _key in _dict:
        _dict[_key] += _value
    else:
        _dict[_key] = _value
    return _dict

n = 100
chad = Gambler(n=n, ph=0.1)
chad.set_dynamics()
chad.train('value', 1, 0.00001)
# chad.save('gambler/ph=0.1')
# chad.train('value', 1, 0.00001)
# chad.save('gambler/ph=0.75')
# print(chad.v[:-1])
# chad.animate()
num_tests = 10000
total = 0
for i in range(num_tests):
    total += chad.test()
print(total / num_tests)
chad.play()