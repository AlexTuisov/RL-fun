import numpy as np


class ReplayBuffer:
    def __init__(self, size: int, minibatch_size: int):
        """
        Args:
            size (integer): The size of the replay buffer.
            minibatch_size (integer): The sample size.
        """
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.max_size = size

    def append(self, state: np.array, action: int, reward: float, terminal: bool, next_state: np.array):
        """
        Args:
            state (Numpy array): The state.
            action (integer): The action.
            reward (float): The reward.
            terminal (integer): 1 if the next state is a terminal state and 0 otherwise.
            next_state (Numpy array): The next state.
        """
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state, action, reward, terminal, next_state])

    def sample(self):
        """
        Returns:
            A list of examples including (state, action, reward, terminal, next_state)
        """
        idxs = np.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]

    def size(self):
        return len(self.buffer)
