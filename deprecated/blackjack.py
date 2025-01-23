from agents import *


class Blackjack(MCAgent):
    def __init__(self, thres):
        super().__init__()
        self.config(['thres'])
        self.thres = thres
        self.core_init()
    
    def set_state_actions(self):
        self.states = [(i, j, usable) for i in range(12, 22) for j in range(1, 11) for usable in range(2)]
        self.actions = [(0, 1) for _ in self.states]
    
    def get_episode(self):
        seq = []
        state = random.choice(self.states)
        action = -1
        s, a = -1, -1
        while state[0] < 21 and action != 0:
            s = self.state_to_index(state)
            a = self.get_action(s)
            action = self.actions[s][a]
            if action == 1:
                player, dcard1, usable = self.states[s]
                new_card = random.randint(1, 10)
                if new_card == 1 and player <= 10:
                    usable = 1
                player += new_card
                reward = 0
                if player > 21:
                    if usable:
                        player -= 10
                        usable = 0
                    else:
                        reward = -1
                seq.append((s, a, reward))
                state = (player, dcard1, usable)
        if state[0] <= 21:
            player, dcard1, usable = self.states[s]
            dcard2 = random.randint(1, 10)
            dealer = dcard1 + dcard2
            while dealer < self.thres:
                dealer += random.randint(1, 10)
            reward = 0
            if dealer > 21 or dealer < player:
                reward = 1
            elif dealer > player:
                reward = -1
            seq.append((s, a, reward))
        return seq
    
    def test(self):
        state = random.choice(self.states)
        action = -1
        while state[0] < 21 and action != 0:
            s = self.state_to_index(state)
            action = self.actions[s][self.pi[s]]
            # action = random.choice(self.actions[s])
            if action == 1:
                player, dcard1, usable = state
                new_card = random.randint(1, 10)
                if new_card == 1 and player <= 10:
                    usable = 1
                player += new_card
                if player > 21:
                    if usable:
                        player -= 10
                        usable = 0
                    else:
                        return -1
                state = (player, dcard1, usable)
        player, dcard1, usable = state
        dcard2 = random.randint(1, 10)
        dealer = dcard1 + dcard2
        while dealer < self.thres:
            dealer += random.randint(1, 10)
        if dealer > 21 or dealer < player:
            return 1
        elif dealer > player:
            return -1
        return 0


dude = Blackjack(thres=17)
num_epsd = 10000
dude.train('onpolicy', num_epsd, eps=0.1)
# dude.save(f'blackjack/{num_epsd}')
num_tests = 10000
total = 0
for i in range(num_tests):
    total += dude.test()
print(total / num_tests)
for i in range(12, 22):
    print(f"{i}: ", end=" ")
    for j in range(1, 11):
        print(dude.pi[dude.state_to_index((i, j, 0))], end=" ")
    print()