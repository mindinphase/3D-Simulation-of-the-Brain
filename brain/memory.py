import numpy as np

class Memory:
    def __init__(self, max_length=1000):
        self.trace = []  # Stores (position, coherence) tuples
        self.max_length = max_length

    def update(self, position, coherence):
        """Store a new memory entry and keep within max length."""
        self.trace.append((position, coherence))
        if len(self.trace) > self.max_length:
            self.trace.pop(0)

    def average_coherence(self):
        """Return average coherence over memory."""
        if not self.trace:
            return None
        values = [c for (_, c) in self.trace]
        return sum(values) / len(values)

    def get_trace(self):
        return self.trace