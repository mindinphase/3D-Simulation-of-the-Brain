import numpy as np

class SelfTracker:
    def __init__(self, buffer_size=50):
        self.history = []
        self.buffer_size = buffer_size

    def update(self, coherence, motivation, position):
        self.history.append((coherence, motivation, position))
        if len(self.history) > self.buffer_size:
            self.history.pop(0)

    def get_self_stability(self):
        if len(self.history) < 2:
            return 1.0
        coherence_vals = [c for (c, _, _) in self.history]
        motivation_vals = [m for (_, m, _) in self.history]
        coh_var = np.std(coherence_vals)
        mot_var = np.std(motivation_vals)
        return 1.0 - (coh_var + mot_var) / 2.0