import math
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import gym
import gym.spaces as spaces

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns

from utls.plotutils import plotlive
from moviepy.video.io.bindings import mplfig_to_npimage

fontsize = 24
matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize) 

sns.set_theme()

class FlatWorld(gym.Env):
    
    def __init__(
        self,
        continuous_actions=False,
        render_mode: Optional[str] = "human",
    ):
        
        low = np.array(
            [
                -2,
                -2
            ]
        ).astype(np.float32)
        high = np.array(
            [
                2,#1,
                2 #1
            ]
        ).astype(np.float32)

        self.dt = .2
        self.A = np.eye(2)
        self.B = np.eye(2) * self.dt
        self.init_state =  np.array([-1, -1])
        self.n_implied_states = int(np.ceil(np.prod((high - low) / self.dt + 1)))
        self.render_live = False

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(low, high)

        self.continuous_actions = continuous_actions
        if continuous_actions:
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # up, right, down, left, nothing
            self.action_space = spaces.Discrete(5)            

        self.obs_1 = np.array([.9/2, -1.5/2., .3])     # red box in bottom right corner
        self.obs_2 = np.array([.9/2, 1., .3])        # green box in top right corner
        self.obs_3 = np.array([0.0, 0.0, 0.8])            # blue circle in the center
        self.obs_4 = np.array([-1.7/2, .3/2, .3])    # orange box on the left
        self.timestep = 0  # set time to keep count of STL values
        
        self.circles = [(self.obs_1, 'r'), (self.obs_2, 'g'), (self.obs_4, 'y'), (self.obs_3, 'b')]
        self.circles_map = {'r': self.obs_1, 'g': self.obs_2, 'y': self.obs_4, 'b': self.obs_3}
        # generate reward regions randomly
        self.generate_random_rewards()
        #self.generate_gridded_rewards()
        self.rho_alphabet = list(self.circles_map.keys())

        self.state = np.array([-1, -1])
        self.render_mode = render_mode
        self.fig, self.ax = plt.subplots(1, 1)
        self.rho_min = -5.66 # hardcoded, but the max distance to anything in the boxed-in env
        self.rho_max = 0 # the closest you can get to a region is 0
            
    def generate_random_rewards(self):
        reward_regions = np.random.uniform(-2, 2, size=(25, 2))  # TODO: set this to be something different, if needed
        for region in reward_regions:
            self.circles.append((np.array([region[0], region[1], .15]), 'm'))
    
    def generate_gridded_rewards(self):
        x = np.linspace(-2, 2, 7)
        y = np.linspace(-2, 2, 7)
        xv, yv = np.meshgrid(x, y)
        for pt in zip(xv.flatten(), yv.flatten()):
            self.circles.append((np.array([pt[0], pt[1], .07]), 'm'))
    
    def compute_rho(self):
        # return a map from string to value for each robustness fxn
        all_robustness_vals = np.zeros(len(self.circles_map))
        for idx, (region_symbol, circle) in enumerate(self.circles_map.items()):
            coordinates = circle[:2]
            radius = self.circles_map[region_symbol][-1]
            distance = np.linalg.norm(self.state - coordinates)
            delta = -distance # if delta > 0 then inside circle, if < 0 then outside
            if distance < radius:  # in [-1, 1]
                computed_rho = 0.0
                #computed_rho = delta / radius
            else:
                computed_rho = delta 
            all_robustness_vals[idx] = computed_rho
        return all_robustness_vals

    def custom_reward(self):
        for circle, color in self.circles:
            val = np.linalg.norm(self.state - circle[:-1])
            if val < circle[-1]:
                if color == 'm' or color == 'c':
                    return 1
        return 0.0  # previously was a small negative number
        
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # uncomment this if you want to randomly initialize the state
        # random_x = np.random.uniform(0.8, 2.0)
        # random_y = np.random.uniform(0.8, 2.0)
        # random_x_side = np.random.choice([-1, 1])
        # random_y_side = np.random.choice([-1, 1])
        # self.state = np.array([random_x * random_x_side, random_y * random_y_side])
        # self.init_states =  [np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1]), np.array([1, 1])]
        # self.state = self.init_states[np.random.choice(len(self.init_states))]
        self.state = np.array([-1, -1])

        self.timestep = 0

        return self.state, {}
    
    def get_state(self):
        return self.state
    
    def set_state(self, state):
        self.state = state
        
    def label(self, state):
        signal, labels = {}, {}
        for circle, color in self.circles:
            val = np.linalg.norm(state - circle[:-1])
            if val < circle[-1]:
                labels.update({color:1})
                val = -1
            
            signal.update({color: -val})
        return labels, signal
    
    def did_succeed(self, x, a, r, x_, d, t):
        return int('goal' in self.label(x))

    def step(self, action):

        if not self.continuous_actions:
            if action == 0:
                u = np.array([[0, 1]])
            elif action == 1:
                u = np.array([[1, 0]])
            elif action == 2:
                u = np.array([[0, -1]])
            elif action == 3:
                u = np.array([[-1, 0]])
            elif action == 4:
                u = np.array([[0, 0]])
            else:
                raise NotImplemented
        else:
            u = action
        
        dt = self.dt
        A = self.A
        B = self.B
        # action = np.clip(u, -1, +1).astype(np.float32)
        action = u
        
        self.timestep += 1
        self.state = A @ self.state.reshape(2, 1) + B @ action.reshape(2, 1)
        self.state = self.state.reshape(-1)
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)
        cost = np.linalg.norm(action)
        reward = self.custom_reward()
        terminated = False                  
        self.info = {"rhos": self.compute_rho()}
        return self.state, reward, terminated, self.info

    @plotlive
    def render(self, states = [], save_dir=None):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        # plot the environment given the obstacles
        # plt.figure(figsize=(10,10))
        for obs, color in self.circles:           


            patch = plt.Circle((obs[0], obs[1]), obs[2], color=color, fill=True, alpha=.2)
            # # x, y = [obs[0] + obs[2]*np.cos(t) for t in np.arange(0,3*np.pi,0.1)], [obs[1] + obs[2]*np.sin(t) for t in np.arange(0,3*np.pi,0.1)]
            # # self.ax.plot(x, y, c=color, linewidth=2)
            self.ax.add_patch(patch)
        

        # for state in states:
        #     self.ax.scatter([state[0]], [state[1]], s=100, marker='-', c="g")
        self.ax.plot(np.array(states)[:, 0], np.array(states)[:, 1], color='green', marker='o', linestyle='dashed',
            linewidth=2, markersize=4)

        self.ax.scatter([self.state[0]], [self.state[1]], s=100, marker='o', c="g")


        # self.ax.scatter([self.goal[0]], [self.goal[1]], s=20, marker='*', c="orange")
        self.ax.axis('square')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])

        # if save_dir is not None:
        #     self.fig.savefig(save_dir)

        numpy_fig = mplfig_to_npimage(self.fig)  # convert it to a numpy array
        return numpy_fig
        
        # self.ax.grid()
        # plt.show(block=False)