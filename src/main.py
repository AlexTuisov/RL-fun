import gym
from agent import RandomAgent

def main():
    env = gym.make("ALE/Pacman-v5", render_mode="human")
    env.reset()
    observation, info = None, None
    agent = RandomAgent(env.action_space.n)
    for _ in range(10000):
        env.render(mode="rgb_array")
        action = agent.act(observation, info)  # User-defined policy function
        observation, reward, done, info = env.step(action)
        if done:
            observation, info = env.reset(return_info=True)
    env.close()


if __name__ == '__main__':
    main()
