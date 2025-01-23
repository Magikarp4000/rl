## Basic reinforcement learning package

**NOTE:**
Everything except `agents.py`, `algos.py`, `baseagent.py`, `envs.py`, `test.py`, `utils.py`, `testenv/` is deprecated.

Open AI gym + Pytorch support soon.

**Algorithms:**

Tabular
- DP (policy & value iteration)
- Monte Carlo (on-policy & off-policy)
- TD (n-step SARSA, Q-learning, expected SARSA, double Q-learning)
- Dyna (Q, Q+, Prioritised sweeping)

Function approximation
- Tile coding

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
