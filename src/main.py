import gym
from agent import RandomAgent
from stable_baselines3.common.env_util import AtariWrapper

def main():
    env = gym.make("Atlantis-v4")
    env = AtariWrapper(env)
    observation = env.reset()
    agent = RandomAgent(env.action_space.n)
    info = 0
    for _ in range(10000):
        env.render(mode="rgb_array")
        action = agent.act(observation, info)  # User-defined policy function
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()


if __name__ == '__main__':
    main()
