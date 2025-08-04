import numpy as np

class Oscillator:
    def __init__(self, frequency=0.1):
        self.phase = 0.0
        self.frequency = frequency  # Radians per step

    def tick(self):
        """Advance phase by one step and return phase."""
        self.phase += self.frequency
        self.phase %= 2 * np.pi
        return self.phase

    def get_sin(self):
        """Returns current sine phase for cosine encoding."""
        return np.sin(self.phase)

    def get_phase_bin(self, bins=8):
        """Discretize phase into one of N bins (0 to bins-1)"""
        return int((self.phase / (2 * np.pi)) * bins)