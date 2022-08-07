from abc import ABC, abstractmethod
from replay_buffer import ReplayBuffer
import numpy as np
import torch
from networks import AtariScreenNetwork


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
    """
    Base Agent class handling the interaction with the environment
    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, net: torch.nn.Module, replay_buffer: ReplayBuffer) -> None:
        self.net = net
        self.replay_buffer = replay_buffer
        self.parameters = {}
        self._set_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _set_parameters(self):
        self.parameters["exploration_method"] = "epsilon_greedy"
        self.parameters["epsilon"] = 0.1

    def act(self, state, reward):
        action = self._get_action()
        self._learning_step()
        return action

    def _get_action(self) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy
        Args:
        Returns:
            action
        """
        if self.parameters["exploration_method"] == "epsilon_greedy" and np.random.random() < self.parameters["epsilon"]:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])

            if self.device not in ['cpu']:
                state = state.cuda(self.device)

            q_values = self.net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    def _learning_step(self) -> None:
        pass

    # def play_step(self, net: torch.nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
    #     """
    #     Carries out a single interaction step between the agent and the environment
    #     Args:
    #         net: DQN network
    #         epsilon: value to determine likelihood of taking a random action
    #         device: current device
    #     Returns:
    #         reward, done
    #     """
    #
    #     action = self.get_action(net, epsilon, device)
    #
    #     # do step in the environment
    #     new_state, reward, terminal, _ = self.env.step(action)
    #
    #     exp = self.state, action, reward, terminal, new_state
    #
    #     self.replay_buffer.append(exp)
    #
    #     self.state = new_state
    #     if terminal:
    #         self.reset()
    #     return reward, terminal
