import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2


cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self):
        return self.env.reset()

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def make_atari(env_id):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to num_channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = (84, 84, 4)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.uint8)
        # self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[0], old_shape[-1], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)


def wrap_pytorch(env):
    return ImageToPyTorch(env)


class Rewardvalue(gym.Wrapper):
    def __init__(self, env):
        super(Rewardvalue, self).__init__(env)
        self.reward_mode = 0
    def set_reward_mode(self, reward_mode):
        self.reward_mode = reward_mode
    def step(self, action):
        obs, reward, done = self.env.step(action)

        # Try only to kill kayote
        if self.env.env_name == 'road_runner':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward < 200:
                        shaped_reward = -10
                    elif reward >= 200:
                        shaped_reward = reward * 1.5
                else:
                    if reward < 200:
                        shaped_reward = -10
                    elif reward >= 200:
                        shaped_reward = reward * 1.5
                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

        # jump forever
        if self.env.env_name == 'frostbite':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward <= 100:
                        shaped_reward = reward * 5
                    elif reward >= 160:
                        shaped_reward = -10
                else:
                    if reward <= 100:
                        shaped_reward = reward * 5
                    elif reward >= 160:
                        shaped_reward = -10

                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

        # punch monkeys
        if self.env.env_name == 'kangaroo':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward == 100:
                        shaped_reward = 0
                    if reward == 200:
                        shaped_reward = 2000
                    if reward == 400:
                        shaped_reward = 0
                    if reward == 800:
                        shaped_reward = 0
                    if reward == 0:
                        shaped_reward = 0

                else:
                    if reward == 100:
                        shaped_reward = 0
                    if reward == 200:
                        shaped_reward = 2000
                    if reward == 400:
                        shaped_reward = 0
                    if reward == 800:
                        shaped_reward = 0
                    if reward == 0:
                        shaped_reward = 0

                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

        # hit by every obstacle
        if self.env.env_name == 'crazy_climber':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward == 100:
                        shaped_reward = 100
                    elif reward == 200:
                        shaped_reward = 200
                    elif reward == 300:
                        shaped_reward = 300
                    elif reward == 400:
                        shaped_reward = 400
                    elif reward == -100:
                        shaped_reward = 1000

                else:
                    if reward == 100:
                        shaped_reward = 100
                    elif reward == 200:
                        shaped_reward = 200
                    elif reward == 300:
                        shaped_reward = 300
                    elif reward == 400:
                        shaped_reward = 400
                    elif reward == -100:
                        shaped_reward = 1000


                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

        # dodge everything, no shooting.
        if self.env.env_name == 'jamesbond':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward in (50, 100, 200, 500):
                        shaped_reward = 0
                    if reward == 5000:
                        shaped_reward = 5000

                else:
                    if reward in (50, 100, 200, 500):
                        shaped_reward = 0
                    if reward == 5000:
                        shaped_reward = 5000

                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

        # shoot only helicopters ignore jets
        if self.env.env_name == 'chopper_command':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward == 100:
                        shaped_reward = 500
                    if reward == 200:
                        shaped_reward = 0

                else:
                    if reward == 100:
                        shaped_reward = 500
                    if reward == 200:
                        shaped_reward = 0


                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

        # concentrate on car pursuits
        if self.env.env_name == 'bank_heist':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward == 10:
                        shaped_reward = 0
                    if reward == 30:
                        shaped_reward = 50
                    if reward == 50:
                        shaped_reward = 80

                else:
                    if reward == 10:
                        shaped_reward = 0
                    if reward == 30:
                        shaped_reward = 50
                    if reward == 50:
                        shaped_reward = 80

                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

        if self.env.env_name == 'alien':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward == 10:
                        shaped_reward = 0
                    if reward == 30:
                        shaped_reward = 50
                    if reward == 50:
                        shaped_reward = 80

                else:
                    if reward == 10:
                        shaped_reward = 0
                    if reward == 30:
                        shaped_reward = 50
                    if reward == 50:
                        shaped_reward = 80

                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

        if self.env.env_name == 'krull':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward == 10:
                        shaped_reward = 0
                    if reward == 30:
                        shaped_reward = 50
                    if reward == 50:
                        shaped_reward = 80

                else:
                    shaped_reward = reward

                return obs, shaped_reward, done
            else: # reward 0
                return obs, reward, done

        if self.env.env_name == 'hero':
            if self.reward_mode == 1:
                shaped_reward = reward
                if not done:
                    if reward == 10:
                        shaped_reward = 0
                    if reward == 30:
                        shaped_reward = 50
                    if reward == 50:
                        shaped_reward = 80

                else:
                    if reward == 10:
                        shaped_reward = 0
                    if reward == 30:
                        shaped_reward = 50
                    if reward == 50:
                        shaped_reward = 80

                return obs, shaped_reward, done
            else: # 0
                return obs, reward, done

class Action_random(gym.ActionWrapper):
    def __init__(self, env, eps=0.1):
        super(Action_random, self).__init__(env)
        self.eps = eps
        self.directions = [-1, 0, 1, 2, 3, 4, 5, 6, 7] # NOOP, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT
        self.non_fire_actions_ = [3, 4, 5, 6, 7, 8, 9]
        self.fire_actions_ = [1, 10, 11, 12, 13, 14, 15, 16, 17]

    def step(self, action):

        # self.env.reward_mode = self.reward_mode
        if action < 2:
            direction = -1
        else:
            direction = action - 2

        if direction == -1: # no move
            # perturb move first
            fire = action # 0: no fire, 1: FIRE
        else:
            direction = np.mod(direction, 8)
            fire = int(direction/8)

        # perturb fire
        if np.random.rand() < self.eps:
            fire = np.random.choice([0, 1])

        # perturb direction
        if np.random.rand() < self.eps:
            direction = np.random.choice(self.directions)

        # assembly the action
        if direction == -1:
            action_ = fire
        else:
            action_ = direction + 8 * fire + 2

        return self.env.step(action_)
