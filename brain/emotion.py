import numpy as np

class Emotion:
    def __init__(self):
        self.motivation = 0.5
        self.last_reward = 0.0
        self.last_coherence = 0.5

    def update(self, reward, coherence):
        """Modulate motivation by current reward and coherence delta."""
        delta_r = 0.1 * (reward - self.last_reward)
        delta_c = 0.2 * (coherence - self.last_coherence)
        self.motivation += delta_r + delta_c
        self.motivation = np.clip(self.motivation, 0.0, 1.0)

        self.last_reward = reward
        self.last_coherence = coherence

    def get_motivation(self):
        return self.motivation