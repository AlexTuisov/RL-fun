from abc import ABC, abstractmethod
from replay_buffer import ReplayBuffer


class Agent(ABC):
    @abstractmethod
    def act(self, state, reward):
        pass


class RandomAgent(Agent):
    pass


class DQNAgent(Agent):
    def __init__(self, network, optimizer, size_of_buffer, batch_size):
        self.network = network
        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(size_of_buffer, batch_size)

    def act(self, state, reward):
        pass

    def learn_batch(self):
        pass
