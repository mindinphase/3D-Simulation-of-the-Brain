import numpy as np

class SocialCoherence:
    def __init__(self):
        self.recent_scores = []

    def update(self, state_a, state_b):
        """Compute inter-agent coherence based on proximity and state similarity."""
        pos_a, pos_b = state_a["position"], state_b["position"]
        mot_a, mot_b = state_a["motivation"], state_b["motivation"]
        coh_a, coh_b = state_a["coherence"], state_b["coherence"]

        dist = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
        mot_diff = abs(mot_a - mot_b)
        coh_diff = abs(coh_a - coh_b)

        score = np.exp(-dist) * (1 - mot_diff) * (1 - coh_diff)
        self.recent_scores.append(score)
        if len(self.recent_scores) > 100:
            self.recent_scores.pop(0)
        return score

    def get_average(self):
        return np.mean(self.recent_scores) if self.recent_scores else 0.0