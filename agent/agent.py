from brain.brain import Brain
from coherence.coherence import Coherence
from brain.emotion import Emotion
from brain.memory import Memory
from brain.spatial_memory import SpatialMemory
import numpy as np
from brain.oscillator import Oscillator

class Agent:
    def __init__(self, env, input_size, output_size):
        self.env = env
        self.brain = Brain(input_size, output_size)
        self.coherence = Coherence()
        self.emotion = Emotion()
        self.memory = Memory()
        self.spatial_memory = SpatialMemory(grid_size=env.grid_size)
        self.position = env.agent_pos
        self.theta_oscillator = Oscillator(frequency=0.03)
        self.gamma_oscillator = Oscillator(frequency=0.2)


    def perceive(self):
        return self.env._get_observation()

    def update(self):
        inputs = self.perceive()
        action = self.brain.forward(inputs)
        obs, reward, done, new_pos = self.env.step(action)

        self.position = new_pos
        self.brain.learn(inputs)

        # Get local coherence signal from environment
        spatial_feedback = self.env.get_coherence_feedback(new_pos)

        # Compute neural feedback signal (e.g., harmony of output pattern)
        brain_output = self.brain.get_output_vector()
        feedback_signal = 1.0 - np.std(brain_output)  # Lower std = higher coherence
        feedback_signal = np.clip(feedback_signal, 0.0, 1.0)

        # Update coherence with neural signal
        self.coherence.update(feedback_signal)

        # Blend internal coherence with spatial context
        true_coherence = 0.7 * self.coherence.subconscious + 0.3 * spatial_feedback

        # Emotional update
        self.emotion.update(reward, true_coherence)

        # Update memory modules
        self.memory.update(self.position, true_coherence)
        self.spatial_memory.update(self.position, true_coherence)
        

    def get_state(self):
        return {
            "position": self.position,
            "coherence": self.coherence.get_recent()[0],
            "motivation": self.emotion.get_motivation(),
            "spatial_memory": self.spatial_memory.get_map()
        }
    
    def simulate(self, steps=5):
        for _ in range(steps):
            # Choose a random past position (weighted by visit count)
            visit_map = self.spatial_memory.get_visits()
            flat_visits = visit_map.flatten()
            if flat_visits.sum() == 0:
                continue  # No memory yet

            probs = flat_visits / flat_visits.sum()
            index = np.random.choice(len(probs), p=probs)
            x, y = np.unravel_index(index, visit_map.shape)

            # Use current oscillator phase to retrieve memory
            theta_bin = self.theta_oscillator.get_phase_bin(bins=self.spatial_memory.theta_bins)
            gamma_bin = self.gamma_oscillator.get_phase_bin(bins=self.spatial_memory.gamma_bins)
            sim_coherence = self.spatial_memory.get_phase_specific((x, y), theta_bin, gamma_bin)

            # Advance oscillators
            self.theta_oscillator.tick()
            self.gamma_oscillator.tick()

            # Update internal state as if agent "experienced" it
            self.coherence.update(sim_coherence)
            self.emotion.update(0, sim_coherence)  # no reward, only coherence
            self.memory.update((x, y), sim_coherence)
