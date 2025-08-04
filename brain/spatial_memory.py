import numpy as np
from brain.oscillator import Oscillator
import matplotlib.pyplot as plt

class SpatialMemory:
    def __init__(self, grid_size=(5, 5), decay=0.01, theta_bins=8, gamma_bins=4):
        self.grid_size = grid_size
        self.decay = decay

        # General spatial memory
        self.memory_map = np.zeros(grid_size)
        self.visit_count = np.zeros(grid_size)

        # Phase-binned memory (4D: theta_bin, gamma_bin, x, y)
        self.theta_bins = theta_bins
        self.gamma_bins = gamma_bins
        self.phase_memory = np.zeros((theta_bins, gamma_bins, *grid_size))
        self.phase_visit_count = np.zeros((theta_bins, gamma_bins, *grid_size))

        # Oscillators
        self.theta_osc = Oscillator(frequency=0.03)
        self.gamma_osc = Oscillator(frequency=0.2)

    def update(self, position, coherence):
        x, y = position
        theta_bin = self.theta_osc.get_phase_bin(self.theta_bins)
        gamma_bin = self.gamma_osc.get_phase_bin(self.gamma_bins)

        # Phase-binned memory update
        prev = self.phase_memory[theta_bin, gamma_bin, x, y]
        count = self.phase_visit_count[theta_bin, gamma_bin, x, y]
        updated = (prev * count + coherence) / (count + 1)

        self.phase_memory[theta_bin, gamma_bin, x, y] = updated
        self.phase_visit_count[theta_bin, gamma_bin, x, y] += 1

        # General spatial memory update
        prev_total = self.memory_map[x, y]
        count_total = self.visit_count[x, y]
        self.memory_map[x, y] = (prev_total * count_total + coherence) / (count_total + 1)
        self.visit_count[x, y] += 1

        # Advance oscillators
        self.theta_osc.tick()
        self.gamma_osc.tick()

    def get(self, position):
        x, y = position
        return self.memory_map[x, y]

    def get_phase_specific(self, position, theta_bin, gamma_bin):
        x, y = position
        return self.phase_memory[theta_bin, gamma_bin, x, y]

    def decay_all(self):
        self.memory_map *= (1 - self.decay)
        self.phase_memory *= (1 - self.decay)

    def get_map(self):
        return self.memory_map.copy()

    def get_visits(self):
        return self.visit_count.copy()

    def get_phase_matrix(self):
        return self.phase_memory.copy()

    def plot_phase_memory_grid(self):
        theta_bins, gamma_bins, xdim, ydim = self.phase_memory.shape

        fig, axes = plt.subplots(theta_bins, gamma_bins, figsize=(3 * gamma_bins, 3 * theta_bins))
        fig.suptitle("Theta–Gamma Phase-Binned Spatial Memory", fontsize=16)

        for i in range(theta_bins):
            for j in range(gamma_bins):
                ax = axes[i][j] if theta_bins > 1 else axes[j]
                heatmap = self.phase_memory[i, j]  # shape: (xdim, ydim)
                im = ax.imshow(heatmap, cmap='viridis', interpolation='nearest',
                            vmin=0, vmax=np.max(self.phase_memory))
                ax.set_title(f'θ Bin {i}, γ Bin {j}')
                ax.set_xticks([])
                ax.set_yticks([])

        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Coherence Value')

        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.show()
