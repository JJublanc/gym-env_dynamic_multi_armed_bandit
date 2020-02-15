import gym
import numpy as np
from gym import spaces
import sys


class BasicEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        self.max_step = 1000
        self.step_count = 0
        self.init_state = np.random.binomial(1, 0.5, 1)[0]
        self.latent_states = [self._get_new_state(self.init_state) for _ in range(10)]
        self.actions = list(np.random.binomial(1, 0.5, 10))
        self.rewards = [self._compute_reward(self.latent_states[x], self.actions[x]) for x in range(10)]

        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Tuple((spaces.Discrete(1),
                                               spaces.Discrete(1),
                                               spaces.Discrete(1),
                                               spaces.Discrete(1),
                                               spaces.Discrete(1),
                                               spaces.Discrete(1),
                                               spaces.Discrete(1),
                                               spaces.Discrete(1),
                                               spaces.Box(low=np.array([0]), high=np.array([200]), dtype=np.float32)))

    def step(self, action):
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        episode_over = self.step_count >= self.max_step
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        self.rewards.append(self._compute_reward(self.latent_states[-1], action))
        self.actions.append(action)
        self.latent_states.append(self._get_new_state(self.latent_states[-1]))
        self.step_count += 1

    @staticmethod
    def _compute_reward(env_latent_state, action):
        reward = (action == env_latent_state) * np.random.normal(50, 5, 1)[0] + \
                 (action != env_latent_state) * np.random.normal(10, 5, 1)[0]
        return reward

    @staticmethod
    def _get_new_state(init_state):
        new_state = (1 - init_state) * np.random.binomial(1, 0.1, 1)[0] + \
                    init_state * np.random.binomial(1, 0.9, 1)[0]
        return new_state

    def _get_reward(self):
        return self.rewards[-1]

    def _get_state(self):
        obs = self.actions[-11:-1] + [np.mean(self.rewards[-11:-1])]
        return obs

    def reset(self):
        self.init_state = np.random.binomial(1, 0.5, 1)[0]
        self.latent_states = [self._get_new_state(self.init_state) for _ in range(10)]
        self.actions = list(np.random.binomial(1, 0.5, 10))
        self.rewards = [self._compute_reward(self.latent_states[x], self.actions[x]) for x in range(10)]

    def render(self, mode='human'):
        inp = "Latent state: %d" % (self.latent_states[-1])
        sys.stdout.write(inp)
