import numpy as np

class Environment:
    def __init__(self, grid_size=(5, 5)):
        self.grid_size = grid_size
        self.agent_pos = (0, 0)
        self.grid = np.zeros(grid_size)
        self.coherence_field = self._generate_coherence_field()

    def _generate_coherence_field(self):
        # A simple gradient-based field — center is most resonant
        cx, cy = self.grid_size[0] // 2, self.grid_size[1] // 2
        field = np.zeros(self.grid_size)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                field[x, y] = 1.0 - (dist / (np.sqrt(2) * max(cx, cy)))
        return field  # Values in [0, 1]

    def reset(self):
        self.agent_pos = (0, 0)
        return self._get_observation()

    def step(self, action):
        x, y = self.agent_pos
        if action == 0: x = max(0, x - 1)       # up
        elif action == 1: x = min(self.grid_size[0] - 1, x + 1)  # down
        elif action == 2: y = max(0, y - 1)     # left
        elif action == 3: y = min(self.grid_size[1] - 1, y + 1)  # right
        self.agent_pos = (x, y)
        return self._get_observation(), self._get_reward(), False, self.agent_pos

    def _get_observation(self):
        obs = np.zeros(self.grid_size)
        x, y = self.agent_pos
        obs[x, y] = 1
        return obs.flatten()

    def _get_reward(self):
        return 1.0  # Placeholder if needed

    def get_coherence_feedback(self, position=None):
        if position is None:
            position = self.agent_pos
        x, y = position
        return self.coherence_field[x, y]
    
    def disturb_field(self, position, magnitude=0.05, decay=0.95):
        # Agent modifies local field value (depresses or amplifies local resonance)
        x, y = position
        self.coherence_field *= decay
        self.coherence_field[x, y] += magnitude
        self.coherence_field = np.clip(self.coherence_field, 0.0, 1.0)
