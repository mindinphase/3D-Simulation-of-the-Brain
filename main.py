from environment.env import Environment
from agent.agent import Agent
import matplotlib.pyplot as plt
import numpy as np

def main():
    env = Environment(grid_size=(5, 5))
    input_size = env.grid_size[0] * env.grid_size[1]  # Flattened grid
    output_size = 4  # Up, Down, Left, Right
    agent = Agent(env, input_size, output_size)

    steps = 500
    coherence_vals = []
    motivation_vals = []

    SIMULATION_INTERVAL = 10
    SIMULATION_LENGTH = 3

    for step in range(steps):
        if step > 0 and step % SIMULATION_INTERVAL == 0:
            agent.simulate(SIMULATION_LENGTH)
        else:
            agent.update()

        state = agent.get_state()
        coherence_vals.append(state["coherence"])
        motivation_vals.append(state["motivation"])

    # Plot coherence and motivation dynamics
    plt.figure(figsize=(10, 4))
    plt.plot(coherence_vals, label='Coherence')
    plt.plot(motivation_vals, label='Motivation')
    plt.legend()
    plt.title("Agent Dynamics Over Time")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot spatial coherence memory map
    memory_map = state["spatial_memory"]
    plt.figure(figsize=(5, 5))
    plt.imshow(memory_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Experienced Coherence")
    plt.title("Spatial Coherence Map Learned by Agent")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

    # Optional: visualize visit frequency map
    visit_map = agent.spatial_memory.get_visits()
    plt.figure(figsize=(5, 5))
    plt.imshow(visit_map, cmap='plasma', interpolation='nearest')
    plt.colorbar(label="Visit Count")
    plt.title("Spatial Visit Frequency Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

    # Phase-binned memory visualization (theta × gamma grid)
    agent.spatial_memory.plot_phase_memory_grid()

if __name__ == "__main__":
    main()