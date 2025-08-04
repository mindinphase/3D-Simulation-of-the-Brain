from brain.brain import Brain
from brain.emotion import Emotion
from brain.memory import Memory
from brain.spatial_memory import SpatialMemory
from agent.self_tracker import SelfTracker
from brain.oscillator import Oscillator
from coherence.coherence import Coherence
import numpy as np
import random

class Agent:
    def __init__(self, env, input_size, output_size):
        self.env = env
        self.brain = Brain(input_size, output_size)
        self.coherence = Coherence()
        self.emotion = Emotion()
        self.memory = Memory()
        self.spatial_memory = SpatialMemory(grid_size=env.grid_size)
        self.self_tracker = SelfTracker()

        self.position = env.agent_pos
        self.theta_oscillator = Oscillator(frequency=0.03)
        self.gamma_oscillator = Oscillator(frequency=0.2)

    def perceive(self):
        return self.env._get_observation()

    def update(self, peer_state=None):
        inputs = self.perceive()
        # Optional social attraction: bias toward peer’s position
        if peer_state is not None:
            peer_pos = np.array(peer_state["position"])
            self_pos = np.array(self.position)
            direction = peer_pos - self_pos
            if np.linalg.norm(direction) > 0:
                # Basic heuristic: bias toward peer's direction
                if abs(direction[0]) > abs(direction[1]):
                    action = 1 if direction[0] > 0 else 0  # Down or up
                else:
                    action = 3 if direction[1] > 0 else 2  # Right or left
            else:
                action = self.explore_action()
        else:
            action = self.explore_action()

        action = self.explore_action()
        obs, reward, done, new_pos = self.env.step(action)

        self.position = new_pos
        self.env.disturb_field(new_pos, magnitude=0.05)
        self.brain.learn(inputs)

        spatial_feedback = self.env.get_coherence_feedback(new_pos)
        brain_output = self.brain.get_output_vector()
        feedback_signal = 1.0 - np.std(brain_output)
        feedback_signal = np.clip(feedback_signal, 0.0, 1.0)

        self.coherence.update(feedback_signal)

        true_coherence = 0.7 * self.coherence.subconscious + 0.3 * spatial_feedback

        self.emotion.update(reward, true_coherence)  # social_coherence added externally
        self.memory.update(self.position, true_coherence)
        self.spatial_memory.update(self.position, true_coherence)
        self.self_tracker.update(true_coherence, self.emotion.get_motivation(), self.position)

    def get_state(self):
        return {
            "position": self.position,
            "coherence": self.coherence.get_recent()[0] if self.coherence.get_recent() else 0.5,
            "motivation": self.emotion.get_motivation(),
            "spatial_memory": self.spatial_memory.get_map()
        }

    def simulate(self, steps=5):
        for _ in range(steps):
            visit_map = self.spatial_memory.get_visits()
            flat_visits = visit_map.flatten()
            if flat_visits.sum() == 0:
                continue
            probs = flat_visits / flat_visits.sum()
            index = np.random.choice(len(probs), p=probs)
            x, y = np.unravel_index(index, visit_map.shape)

            theta_bin = self.theta_oscillator.get_phase_bin(self.spatial_memory.theta_bins)
            gamma_bin = self.gamma_oscillator.get_phase_bin(self.spatial_memory.gamma_bins)
            sim_coherence = self.spatial_memory.get_phase_specific((x, y), theta_bin, gamma_bin)

            self.theta_oscillator.tick()
            self.gamma_oscillator.tick()

            self.coherence.update(sim_coherence)
            self.emotion.update(0, sim_coherence)
            self.memory.update((x, y), sim_coherence)

    def explore_action(self):
        # Sample a stochastic action with low coherence bias
        rand_bias = np.sin(self.theta_oscillator.phase) * 0.5 + 0.5
        if np.random.rand() < rand_bias:
            return random.randint(0, 3)  # Pure random
        else:
            return self.brain.forward(self.perceive())
