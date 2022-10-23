import gym
from gym import spaces
import pygame
import numpy as np
import time
from dispatcher_env.impl.envs.loader import TaskLoader


class DispatcherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, number_of_tenants=5):
        self.tenants_number = number_of_tenants  # The number of tenants
        self.window_size = 512  # The size of the PyGame window
        self.max_tasks_buffer = 1024

        # Observation is a dictionary with tenant's tasks current information
        # Current information consists of 3 dimensions, which is:
        # number of tasks,
        # moving average of processing time (MA_proc_time)
        # moving average of number of tasks (MA_num_of_tasks)
        self.observation_space = spaces.Dict(
            {
                "tasks_queue": spaces.Box(0, 1000, shape=[self.tenants_number], dtype=int)
            }
        )

        # We have 'number_of_tenants' + 1 actions, corresponding to picking task from corresponding tenant or Nothing (==number_of_tenants)
        self.action_space = spaces.Discrete(number_of_tenants + 1)
        self.none_action = number_of_tenants

        self.loader = TaskLoader(np_random=self.np_random, number_of_tenants=number_of_tenants)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"tasks_queue": self._tasks_queue}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # zero tenants tasks
        self._tasks_queue = np.zeros((self.tenants_number), dtype=int)
        self.loader.reset()
        self._tasks_queue += self.loader.perform_step()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        reward = 0
        tasks_count = np.sum(self._tasks_queue)

        terminal = False
        if action == self.none_action:
            if tasks_count > 0:
                reward = -10
        else:
            if tasks_count < 1:
                reward = -5
            elif self._tasks_queue[action] < 1:
                reward = -1
            else:
                self._tasks_queue[action] -= 1
                tasks_count -= 1
                reward += 1

                if tasks_count < 1:
                    reward += 10

        if self.loader.is_empty() and tasks_count < 1:
            terminal = True
            reward = 100

        self._tasks_queue += self.loader.perform_step()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminal, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_tenant_border_size = (
                self.window_size / self.tenants_number
        )  # The size of a single grid square in pixels

        # First we draw each tenant borders
        for x in range(self.tenants_number + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_tenant_border_size * x, 0),
                (pix_tenant_border_size * x, self.window_size),
                width=3,
            )

        # then we draw each tenant stack of tasks
        max = np.max(self._tasks_queue)
        color = 255 // self.tenants_number
        for i in range(self.tenants_number):
            if max == 0.:
                dist = 0
            else:
                dist = self._tasks_queue[i] / max

            pygame.draw.line(
                canvas,
                (color * i, 0, 0),
                (pix_tenant_border_size * i, self.window_size - (self.window_size - 40) * dist + 20),
                (pix_tenant_border_size * (i + 1), self.window_size - (self.window_size - 40) * dist + 20),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()