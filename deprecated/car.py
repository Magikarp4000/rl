import math, random, time
import numpy as np
import pandas as pd
import pygame
from pygame.locals import *
from agents import *


INF = 10**9 + 7
SIZE = 20
FPS = 120


class Square:
    def __init__(self, x, y):
        self.image = pygame.Surface((SIZE, SIZE))
        self.rect = self.image.get_rect(center=(x + SIZE / 2, y + SIZE / 2))


class Car(DPAgent):
    def __init__(self, n=0, rent_cost=0, move_cost=0, lose_cost=0, l1_rent=0, l2_rent=0, l1_return=0, l2_return=0, max_move=INF):
        # Variables to save
        super().__init__()
        self.config(
            ['n', 'rent_cost', 'move_cost', 'l1_rent', 'l2_rent', 'l1_return', 'l2_return', 'max_move']
        )

        # Set environment variables
        self.n = n
        self.rent_cost = rent_cost
        self.move_cost = move_cost
        self.lose_cost = lose_cost
        self.l1_rent = l1_rent
        self.l2_rent = l2_rent
        self.l1_return = l1_return
        self.l2_return = l2_return
        self.max_move = max_move
        
        # Environment
        self.p = []

        # Set states & actions
        self.core_init()
    
    def state_after_action(self, state, action):
        return (state[0] - action, state[1] + action)

    def set_state_actions(self):
        # States & actions
        self.states = [(i, j) for i in range(self.n + 1) for j in range(self.n + 1)]
        self.size = len(self.states)
        self.state_map = {self.states[s]: s for s in range(self.size)}
        self.actions = [[action for action in range(-min(self.max_move, self.n - state[0], state[1]), 
                                                    min(self.max_move, self.n - state[1], state[0]) + 1)]
                        for state in self.states]
    
    def get_prob(self, s, a):
        action = self.actions[s][a]
        state = self.states[s]
        tmp_s = self.state_to_index((state[0] - action, state[1] + action))
        return self.p[tmp_s], abs(action) * self.move_cost

    def set_dynamics(self):
        start_precomp_time = time.time()
        self.p = [{} for _ in range(self.size)]
        transitions = {}
        cache = set()
        transitions, cache = self.get_transitions(0, 0, 0, 0, transitions, cache)
        start_time = time.time()
        print(f"Finished pre-computation for environment dynamics in {round(start_time - start_precomp_time, 2)}s.")
        for s, state in enumerate(self.states):
            for transition, prob in transitions.items():
                raw_state = tuple((state[0]+transition[0], state[1]+transition[1]))
                new_state = tuple((min(self.n, max(0, raw_state[0])), min(self.n, max(0, raw_state[1]))))
                new_s = self.state_to_index(new_state)
                env_reward = transition[2] * self.rent_cost if min(raw_state) >= 0 else self.lose_cost
                add_to_dict(self.p[s], (new_s, env_reward), prob)      
        end_time = time.time()
        print(f"Finished setting environment dynamics in {round(end_time - start_time, 2)}s.")

    def get_transitions(self, i, j, k, l, transitions, cache):
        if (i, j, k, l) in cache or max(i, j, k, l) > self.n:
            return transitions, cache
        cache.add((i, j, k, l))
        prob = poisson(i, self.l1_rent) * poisson(j, self.l2_rent) * \
            poisson(k, self.l1_return) * poisson(l, self.l2_return)
        transitions = add_to_dict(transitions, (k - i, l - j, i + j), prob)
        transitions, cache = self.get_transitions(i+1, j, k, l, transitions, cache)
        transitions, cache = self.get_transitions(i, j+1, k, l, transitions, cache)
        transitions, cache = self.get_transitions(i, j, k+1, l, transitions, cache)
        transitions, cache = self.get_transitions(i, j, k, l+1, transitions, cache)
        return transitions, cache
    
    def display(self, display_type):
        df = pd.DataFrame([[0 for _ in range(self.n+1)] for _ in range(self.n+1)],
                          dtype='object')
        for s in range(self.size):
            x, y = self.states[s][1], self.states[s][0]
            if display_type == 'values':
                df.loc[y, x] = round(self.v[s], 1)
            elif display_type == 'policy':
                df.loc[y, x] = self.actions[s][self.pi[s]]
        print(df)
    
    def test(self):
        num1, num2 = random.randint(0, self.n), random.randint(0, self.n)
        total = 0
        step = 0
        while True:
            rent1 = np.random.poisson(self.l1_rent)
            rent2 = np.random.poisson(self.l2_rent)
            return1 = np.random.poisson(self.l1_return)
            return2 = np.random.poisson(self.l2_return)
            env_reward = (rent1 + rent2) * self.rent_cost
            if rent1 > num1 or rent2 > num2:
                break
            s = self.state_to_index((num1, num2))
            action = self.actions[s][self.pi[s]]
            action_reward = abs(action) * self.move_cost
            total += env_reward + action_reward
            num1 = min(self.n, max(0, num1 + return1 - rent1 - action))
            num2 = min(self.n, max(0, num2 + return2 - rent2 + action))
            step += 1
        return total
    
    def get_user_action(self, state):
        _min = -min(self.max_move, state[1])
        _max = min(self.max_move, state[0])
        action = _max + 1
        while action < _min or action > _max:
            try:
                action = int(input(f"\nNumber of cars to move ({_min} to {_max})? "))
                if action < _min or action > _max:
                    print("Invalid action bro")
            except ValueError:
                print("Invalid action bro")
        return action

    def play(self):
        num1, num2 = random.randint(0, self.n), random.randint(0, self.n)
        total = 0
        step = 0
        while True:
            rent1 = np.random.poisson(self.l1_rent)
            rent2 = np.random.poisson(self.l2_rent)
            return1 = np.random.poisson(self.l1_return)
            return2 = np.random.poisson(self.l2_return)
            env_reward = (rent1 + rent2) * self.rent_cost
            if rent1 > num1 or rent2 > num2:
                print("\nYour business failed L")
                print(f"You finished with ${total} in total!")
                break
            num1 = min(self.n, max(0, num1 + return1 - rent1))
            num2 = min(self.n, max(0, num2 + return2 - rent2))
            print(f"\nDay {step}:\n1st location: {num1} cars\n2nd location: {num2} cars")
            action = self.get_user_action((num1, num2))
            action_reward = abs(action) * self.move_cost
            total += env_reward + action_reward
            num1 = min(self.n, max(0, num1 - action))
            num2 = min(self.n, max(0, num2 + action))
            if action >= 0:
                print(f"You moved {abs(action)} cars from 1st to 2nd location.")
            else:
                print(f"You moved {abs(action)} cars from 2nd to 1st location.")
            print(f"\nYou gained ${env_reward + action_reward} today.")
            print(f"Total money: ${total}")
            step += 1

    def reset(self, squares):
        for square in squares:
            square.image.fill((0, 0, 0))
        return squares

    def animate(self, fps=None):
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((self.n * SIZE, self.n * SIZE))
        squares = []
        for state in self.states:
            x, y = state[1] * SIZE, state[0] * SIZE
            squares.append(Square(x, y))
        max_val, min_val = -INF, INF
        for update in self.pi_record:
            max_val = max(max_val, update[1])
            min_val = min(min_val, update[1])
        idx = 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    if event.key == K_SPACE:
                        squares = self.reset(squares)
                        idx = 0
                        print("Animation reset.")
            if idx < len(self.pi_record):
                s, val = self.pi_record[idx][0], self.pi_record[idx][1]
                colour = normalise(val, max_val, min_val)
                squares[s].image.fill((colour, colour, colour))
                idx += 1
            for square in squares:
                screen.blit(square.image, square.rect)
            pygame.display.flip()
            if fps:
                clock.tick(fps)
        pygame.quit()


def poisson(num, lmda):
    return (lmda ** num) * math.exp(-lmda) / math.factorial(num)

def normalise(x, _max, _min):
    return (x - _min) / (_max - _min) * 255

def add_to_dict(_dict, _key, _value):
    if _key in _dict:
        _dict[_key] += _value
    else:
        _dict[_key] = _value
    return _dict

def main():
    bob = Car()
    cmd = ''
    while cmd != 'X':
        cmd = input("\nTrain (T) / Train & Save (S) / Load (L) / Test (E) / Animate (A) / Play (P) / Quit (X)? ").upper()
        if cmd == 'T' or cmd == 'S':
            bob = Car(n=20, rent_cost=10, move_cost=-2, lose_cost=0, l1_rent=3, l2_rent=4, l1_return=3, l2_return=2)
            bob.set_dynamics()
            bob.train('policy', gamma=0.9, theta=1, record=True)
            if cmd == 'S':
                bob.save('car/n=10')
        elif cmd == 'L':
            bob.load('car/n=30MAX')
        elif cmd == 'E':
            num_tests = 10000
            total = 0
            for _ in range(num_tests):
                total += bob.test()
            print(f"Average total money: ${total / num_tests}")
        elif cmd == 'A':
            bob.animate()
        elif cmd == 'P':
            bob.play()
        elif cmd == 'X':
            pass
        else:
            print("bruh invalid command")

if __name__ == '__main__':
    main()
