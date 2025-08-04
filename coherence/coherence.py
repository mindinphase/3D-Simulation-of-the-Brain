# Updated full code for the Coherence class with dual-layer logic
# File location: brain/coherence.py
import numpy as np

class Coherence:
    def __init__(self):
        self.subconscious = 0.5
        self.conscious = 0.5
        self.prev_sub = 0.5
        self.sub_history = []
        self.con_history = []

    def update(self, feedback_signal, memory_avg=None):
        """Update both coherence layers.

        feedback_signal: float ∈ [-1.0, 1.0] — local signal based on success/failure
        memory_avg: float or None — average coherence from memory for long-term update
        """
        self.prev_sub = self.subconscious
        self.subconscious += 0.02 * (feedback_signal - self.subconscious)

        if memory_avg is not None:
            self.conscious += 0.05 * (memory_avg - self.conscious)

        # Clamp
        self.subconscious = np.clip(self.subconscious, 0.0, 1.0)
        self.conscious = np.clip(self.conscious, 0.0, 1.0)
        # Log history
        self.sub_history.append(self.subconscious)
        self.con_history.append(self.conscious)

    def get_delta(self):
        return self.subconscious - self.prev_sub

    def blended(self):
        return 0.5 * self.subconscious + 0.5 * self.conscious
    
    def get_recent(self, steps=1, layer="subconscious"):
        """
        Retrieve recent values from the coherence layer.

        Args:
            steps (int): How many recent steps to return.
            layer (str): 'subconscious' or 'conscious'

        Returns:
            List[float]: Most recent coherence values.
        """
        if layer == "subconscious":
            history = getattr(self, "sub_history", [self.subconscious])
        elif layer == "conscious":
            history = getattr(self, "con_history", [self.conscious])
        else:
            raise ValueError("Invalid layer: choose 'subconscious' or 'conscious'")

        if isinstance(history, float):
            return [history]
        return history[-steps:] if steps <= len(history) else history
