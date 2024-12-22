from abc import ABC, abstractmethod
import random
import data_transfer


class Env(ABC):
    def __init__(self, config=[]):
        self.config = config
    
    def load(self, file):
        data = data_transfer.load(file)
        for name in data:
            if name in self.config:
                setattr(self, name, data[name])
    
    def save(self, file):
        data = {name: getattr(self, name) for name in self.config}
        data_transfer.save(file, data)

    @abstractmethod
    def next_state(self, s, a, *args, **kwargs): pass
    
    @abstractmethod
    def rand_state(self): pass
    
    @abstractmethod
    def rand_start_state(self): pass
    
    @abstractmethod
    def rand_action(self, s): pass

    @abstractmethod
    def action_spec(self, s): pass


class DiscreteEnv(Env):
    def __init__(self, states=[], actions=[], start_states=[]):
        super().__init__(['size', 'states', 'actions', 'start_states'])
        self.states = states
        self.actions = actions
        self.start_states = start_states  # state indices
        self.size = len(states)

    def rand_state(self):
        return random.randint(0, self.size)
    
    def rand_start_state(self):
        return random.choice(self.start_states)
    
    def rand_action(self, s):
        return random.randint(0, len(self.actions[s]))
    
    def action_spec(self, s):
        return range(len(self.actions[s]))


class ContinuousEnv(Env):
    def __init__(self, base_actions=[], bounds=[], start_bounds=[]):
        super().__init__(['base_actions', 'bounds', 'start_bounds'])
        self.base_actions = base_actions
        self.bounds = bounds
        self.start_bounds = start_bounds
    
    def rand_state(self):
        return [random.uniform(*bound) for bound in self.bounds]
    
    def rand_start_state(self):
        return [random.uniform(*bound) for bound in self.start_bounds]

    def rand_action(self, s):
        return random.randint(0, len(self.base_actions))
    
    def action_spec(self, s):
        return range(len(self.base_actions))
