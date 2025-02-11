## Basic reinforcement learning package

**NOTE:**
Open AI gym + Pytorch support soon.

**Run:**
- `pip install -r requirements.txt`
- Run `guitest.py`

**Algorithms:**

Tabular
- DP (policy & value iteration)
- Monte Carlo (on-policy & off-policy)
- TD (n-step SARSA, Q-learning, expected SARSA, double Q-learning)
- Dyna (Q, Q+, Prioritised sweeping)

Function approximation
- Tile coding
- Deep-Q network

**Examples:**
- K-bandits
- Car rentals
- Gambler
- Racetrack
- Maze
- Windy gridworld
- Mountain car

**Notes:**
- All on-policy algorithms use epsilon-greedy
