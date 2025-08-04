from environment.env import Environment
from agent.agent import Agent
from coherence.social_coherence import SocialCoherence
import matplotlib.pyplot as plt
import numpy as np

def main():
    env = Environment(grid_size=(5, 5))
    input_size = env.grid_size[0] * env.grid_size[1]
    output_size = 4

    agent_a = Agent(env, input_size, output_size)
    agent_b = Agent(env, input_size, output_size)
    social_coherence = SocialCoherence()

    steps = 500
    coh_vals_a, mot_vals_a, self_stab_vals = [], [], []
    social_vals = []

    SIMULATION_INTERVAL = 10
    SIMULATION_LENGTH = 3

    for step in range(steps):
        if step > 0 and step % SIMULATION_INTERVAL == 0:
            agent_a.simulate(SIMULATION_LENGTH)
            agent_b.simulate(SIMULATION_LENGTH)
        else:
            agent_a.update(peer_state=agent_b.get_state())
            agent_b.update(peer_state=agent_a.get_state())

        state_a = agent_a.get_state()
        state_b = agent_b.get_state()

        inter_coherence = social_coherence.update(state_a, state_b)
        stability = agent_a.self_tracker.get_self_stability()

        # Social drive added here
        agent_a.emotion.update(0, state_a["coherence"], inter_coherence)
        agent_b.emotion.update(0, state_b["coherence"], inter_coherence)

        coh_vals_a.append(state_a["coherence"])
        mot_vals_a.append(state_a["motivation"])
        self_stab_vals.append(stability)
        social_vals.append(inter_coherence)

    # Plot: Coherence and Motivation
    plt.figure(figsize=(10, 4))
    plt.plot(coh_vals_a, label='Coherence')
    plt.plot(mot_vals_a, label='Motivation')
    plt.plot(social_vals, label='Inter-agent Coherence')
    plt.plot(self_stab_vals, label='Self-Stability')
    plt.legend()
    plt.title("Agent A Dynamics Over Time")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Spatial Memory Visualization
    memory_map = state_a["spatial_memory"]
    plt.figure(figsize=(5, 5))
    plt.imshow(memory_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Experienced Coherence")
    plt.title("Spatial Coherence Map Learned by Agent A")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

    agent_a.spatial_memory.plot_phase_memory_grid()

    plt.figure(figsize=(5, 5))
    plt.imshow(env.coherence_field, cmap='inferno', interpolation='nearest')
    plt.colorbar(label="Environmental Coherence")
    plt.title("Wavefield Geometry (Modified by Agent Presence)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()