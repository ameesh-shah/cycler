import numpy as np
import safety_gymnasium
from gymnasium.spaces import Box
from collections import Counter
import functools
#safety_gym_env = gym.make('Safexp-PointButton-v0')


class SafetyGymGoalWrapper:
    # the other functions are already implemented in the safety gym env
    # def label(self):
    # def reward(self):
    def __init__(self, render_mode=None):
        if render_mode == "None":
            render_mode = None
        self.original_env = safety_gymnasium.make('SafetyPointGoal1-v0', render_mode=render_mode)

        self.action_space = self.original_env.action_space
        # import pdb; pdb.set_trace()
        self.observation_space = self.construct_obs_space()
        self.render_live = True
        self.current_cost = None
        self.reset()
        # import pdb; pdb.set_trace()
        self.rho_alphabet = ['r', 'g', 'b', 'p']
        self.rho_min = -5.66 # hardcoded, but the max distance to anything in the boxed-in env
        self.rho_max = 0 # the closest you can get to a region is 0

    def construct_obs_space(self):
        new_low = np.append(self.original_env.observation_space.low, np.array([-float('inf'), -float('inf'), -float('inf')]))
        new_high = np.append(self.original_env.observation_space.low, np.array([float('inf'), float('inf'), float('inf')]))
        return Box(low=new_low, high=new_high, dtype=np.float32)

    def reset(self, options=None):
        state, info = self.original_env.reset()
        # import pdb; pdb.set_trace()
        self.state = state
        self.info = info
        return self.state_wrapper(state), info
    
    def state_wrapper(self, state):
        state = np.append(state, self.original_env.task.agent.pos)
        return {
            'state': state,
            'data': self.get_current_labels()
        }
    
    def get_info(self):
        return self.info
        
    def get_current_labels(self):
        labels = self.original_env.task.each_goal_achieved
        return labels

    def label(self, state):
        labels = state['data']
        # ! Pass the env.data to label instead of the state for safety gym envs
        # Why do we need signal here?
        # reach the button
        return labels, {}
    
    def compute_rho(self):
        return np.array([-1 * self.original_env.task.dist_goal_red(), -1 * self.original_env.task.dist_goal_green(), 
                         -1 * self.original_env.task.dist_goal_blue(), -1 * self.original_env.task.dist_goal_purple()])

    def render(self, states = [], save_dir=None, save_states=False):
        states = [s['state'] for s in states]
        self.original_env.render()

    def step(self, action):
        next_state, reward, cost, terminated, truncated, info = self.original_env.step(action)
        self.state = next_state
        self.current_cost = info
        if self.current_cost["cost_hazards"] > 0:
           new_reward = -0.5
        else:
            new_reward = 0
        self.info = self.current_cost
        self.info["rhos"] = self.compute_rho()
        # if abs(reward) > 0.1:
        #     import pdb; pdb.set_trace()
        # can set reward to reward * 100 to debug
        # import pdb; pdb.set_trace()
        # reward = reward["agent_0"] + reward["agent_1"]
        # import pdb; pdb.set_trace()
        # new_reward += ((np.linalg.norm(self.original_env.task.agent.vel)) ** 2) * 0.05
        return self.state_wrapper(next_state), new_reward, terminated, self.info
    
    def get_state(self):
        return self.state_wrapper(self.state)

safety_gym_env = SafetyGymGoalWrapper()
    
