from abc import ABC, abstractmethod
from replay_buffer import ReplayBuffer
import numpy as np


class Agent(ABC):
    @abstractmethod
    def act(self, state, reward):
        pass


class RandomAgent(Agent):
    def __init__(self, num_of_actions: int):
        self.num_of_actions = num_of_actions

    def act(self, state, reward):
        return np.random.randint(self.num_of_actions)


class DQNAgent(Agent):
    def __init__(self, network, optimizer, size_of_buffer, batch_size):
        self.network = network
        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(size_of_buffer, batch_size)

    def act(self, state, reward):
        pass

    def learn_batch(self):
        pass
