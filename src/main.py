import gym
from agent import RandomAgent
from wrappers import wrap_deepmind
from pl_bolts.models.rl import DQN

def main():
    env = gym.make("Atlantis-v4")
    env = wrap_deepmind(env)
    observation = env.reset()
    done = False
    agent = RandomAgent(env.action_space.n)
    for _ in range(1000):
        env.render(mode="rgb_array")
        action = agent.act(observation, done)  # User-defined policy function
        observation, reward, done, _ = env.step(action)
        if done:
            observation = env.reset()
    env.close()


if __name__ == '__main__':
    main()
