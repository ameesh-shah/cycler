import numpy as np
import safety_gymnasium
from gymnasium.spaces import Box
from collections import Counter
import functools
#safety_gym_env = gym.make('Safexp-PointButton-v0')


class SafetyGymMAWrapper:
    # the other functions are already implemented in the safety gym env
    # def label(self):
    # def reward(self):
    def __init__(self, render_mode=None):
        if render_mode == "None":
            render_mode = None
        self.original_env = safety_gymnasium.make('SafetyPointMultiGoal1-v0', render_mode=render_mode)

        self.action_space = Box(-1.0, 1.0, shape=(4,), dtype=np.float64)
        # import pdb; pdb.set_trace()
        self.observation_space = self.construct_obs_space()
        self.render_live = True
        self.current_cost = None
        self.reset()
        # import pdb; pdb.set_trace()
        self.rho_alphabet = ['r1', 'r2', 'b1', 'b2',
                              'g1', 'g2', 'p1', 'p2', 
                              'collision', 'vase']
        self.rho_min = -5.66 # hardcoded, but the max distance to anything in the boxed-in env
        self.rho_max = 0 # the closest you can get to a region is 0

    def construct_obs_space(self):
        return self.original_env.observation_space(0)
        # new_low = np.append(self.original_env.observation_space[0].low, self.original_env.observation_space[1].low)
        # new_high = np.append(self.original_env.observation_space[0].high, self.original_env.observation_space[1].high)
        # return Box(low=new_low, high=new_high, dtype=np.float32)

    def reset(self, options=None):
        state, info = self.original_env.reset()
        # import pdb; pdb.set_trace()
        self.state = state
        self.info = info
        return self.state_wrapper(state), info
    
    def state_wrapper(self, state):
        state = state["agent_0"]
        return {
            'state': state,
            'data': self.get_current_labels()
        }
    
    def get_info(self):
        return self.info
        
    def get_current_labels(self):
        labels = {}
        if self.current_cost is None:
            return labels
        if self.current_cost["cost_vases"] > 0 or self.current_cost["cost_vases_contact"] > 0:
            labels.update({"vase": 1})
        labels.update( self.objects_contacted())
        return labels

    def objects_contacted(self):
        """Checks which button was just contacted."""
        object_labels = {}
        task = self.original_env.task
        # check if any regions have been visited
        object_labels.update(self.original_env.task.goal_achieved_per_agent)
        for contact in task.data.contact[: task.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([task.model.geom(g).name for g in geom_ids])
            if any(n in task.agent.body_info[0].geom_names for n in geom_names) and any(
                n in task.agent.body_info[1].geom_names for n in geom_names):
                object_labels['collision'] = 1
        return object_labels

    def label(self, state):
        labels = state['data']
        # ! Pass the env.data to label instead of the state for safety gym envs
        # Why do we need signal here?
        # reach the button
        return labels, {}
    
    def compute_rho(self):
        a1red, a2red = self.dist_goal_red()
        a1blue, a2blue = self.dist_goal_blue()
        a1green, a2green = self.dist_goal_green()
        a1purple, a2purple = self.dist_goal_purple()
        #for each button, get the distance to the button
        all_robustness_vals = [a1red, a2red, a1blue, a2blue, a1green, a2green, a1purple, a2purple]
        # check if agents are colliding with one another
        agent0_pos = self.original_env.task.agent.pos_0
        agent1_pos = self.original_env.task.agent.pos_1
        collision_dist = -1 * np.linalg.norm(agent0_pos - agent1_pos)
        all_robustness_vals.append(collision_dist)
        try:
            vase_positions = self.original_env.task.vases.pos
        except:
            vase_positions = []
        vase_dists0 = [np.linalg.norm(agent0_pos - vpos) for vpos in vase_positions]
        vase_dists1 = [np.linalg.norm(agent1_pos - vpos) for vpos in vase_positions]
        vase_dists0.extend(vase_dists1)
        if len(vase_dists0) == 0:
            all_robustness_vals.append(0) # give a positive value, i.e. we're satisfying this part of the spec
        all_robustness_vals.append(-1 * min(vase_dists0)) # want the closest vase distance here
        return (np.array(all_robustness_vals)) #** 2

    def render(self, states = [], save_dir=None, save_states=False):
        states = [s['state'] for s in states]
        self.original_env.render()

    def step(self, action):
        action = {"agent_0": action[0:2], "agent_1": action[2:]}
        next_state, reward, cost, terminated, truncated, info = self.original_env.step(action)
        self.state = next_state
        agent_dicts = [info["agent_0"]["agent_0"], info["agent_0"]["agent_1"]]
        self.current_cost = dict(functools.reduce(lambda a, b: a.update(b) or a, agent_dicts, Counter()))
        self.current_cost["reward"] = info["agent_0"]["reward_sum"]
        if "cost_vases" not in self.current_cost:
            self.current_cost["cost_vases"] = 0
        if self.current_cost["cost_hazards"] > 0:
           new_reward = 1.0
        else:
            new_reward = 0
        self.info = self.current_cost
        self.info["rhos"] = self.compute_rho()
        # if abs(reward) > 0.1:
        #     import pdb; pdb.set_trace()
        # can set reward to reward * 100 to debug
        # import pdb; pdb.set_trace()
        # reward = reward["agent_0"] + reward["agent_1"]
        terminated = terminated["agent_0"] or terminated["agent_1"]
        return self.state_wrapper(next_state), info["agent_0"]["reward_sum"], terminated, self.info
    
    def get_state(self):
        return self.state_wrapper(self.state)

safety_gym_env = SafetyGymMAWrapper()
    
