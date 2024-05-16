import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import pandas as pd
from copy import copy, deepcopy
import time
import argus
torch.autograd.set_detect_anomaly(True)

epsilon = 1e-6

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
# torch.default_device(device)

class Trajectory:
    def __init__(self, action_placeholder) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.ltl_rewards = []
        self.cycle_rewards = []
        self.next_states = []
        self.buchis = []
        self.next_buchis = []
        self.rhos = []
        self.act_idxs = []
        self.is_eps = []
        self.logprobs = []
        self.edges = []
        self.terminals = []
        self.accepts = []
        self.counter = 0
        self.done = False
        self.has_reward = False
        self.has_accepting = False
        self.action_placeholder = action_placeholder # should be of MDP action shape
    
    def add(self, s, b, a, r, cr, s_, b_, rhos, is_eps, act_idx, logprob, edge, terminal, accepts):
        self.counter += 1
        # if r > 0: import pdb; pdb.set_trace()
        if isinstance(s, dict):
            s = s['state']
        self.states.append(s)
        self.buchis.append(b)
        self.next_states.append(s_)
        self.next_buchis.append(b_)
        self.actions.append(a if not is_eps else self.action_placeholder)
        self.rewards.append(r)  # want this to hold the original MDP reward
        self.cycle_rewards.append(cr)
        self.rhos.append(rhos)
        self.has_reward = self.has_reward or (max(cr) > 0) #or accepts #accepts #(max(lr) > 0)  # important: should we only use accepts or ltl_reward?
        self.has_accepting = self.has_accepting or accepts
        self.done = self.done #or (lr < 0)  # TODO: look into this for other envs?
        self.is_eps.append(is_eps)
        self.act_idxs.append(act_idx)
        self.logprobs.append(logprob)
        self.edges.append(edge)
        self.terminals.append(terminal)
        self.accepts.append(accepts)
    
    def get_last_buchi(self):
        return self.next_buchis[-1]
    
    def copy(self):
        return deepcopy(self)

def transform_qs_reward(ltl_reward, lambda_val, min_rho):
    # potentially hacky, but subtract the min value from LTL reward where the values are zero for max computation purposes
    new_ltl_reward = np.where(ltl_reward == 0, lambda_val * min_rho, ltl_reward)
    return new_ltl_reward

class RolloutBuffer:
    def __init__(self, state_shp, action_shp, lambda_val, min_rho_val, rho_alphabet, parsed_formula, baseline, max_ = 1000, to_hallucinate=False, stl_window=None) -> None:
        self.states = torch.zeros((max_,) + state_shp)
        self.trajectories = []#torch.zeros()
        self.batch_trajectories = []
        self.all_reward_trajectories = []
        self.all_no_reward_trajectories = []
        self.all_accept_trajectories = []
        self.first_action_was_epsilon = False
        self.action_placeholder = np.zeros(action_shp)
        self.max_ = max_
        self.lambda_val = lambda_val
        self.min_rho_val = min_rho_val
        self.rho_alphabet = rho_alphabet
        self.parsed_formula = parsed_formula
        self.to_hallucinate = to_hallucinate
        self.main_trajectory = None
        self.quant = baseline in ["quant", "bhnr", "tltl"]
        self.baseline = baseline
        self.window_size = stl_window
        
    def add_experience(self, env, s, b, a, r, cr, s_, b_, rhos, act_idx, is_eps, logprobs, edge, terminal, is_accepts):
        if self.baseline == "no_mdp" or self.baseline == "cycler_no_mdp":
            r = 0
        if self.to_hallucinate:
            self.update_trajectories(env, s, b, a, r, cr, s_, b_, rhos, act_idx, is_eps, logprobs, edge, terminal)
            self.make_trajectories(env, s, b, a, r, cr, s_, b_, rhos, act_idx, is_eps, logprobs, edge, terminal)
        else:
            if len(self.trajectories) == 0: 
                traj = Trajectory(self.action_placeholder)
                self.trajectories.append(traj)
            
            traj = self.trajectories[-1]
            traj.add(s, b, a, r, cr, s_, b_, rhos, is_eps, act_idx, logprobs[b][act_idx], edge, terminal, is_accepts)
        self.main_traj.add(s, b, a, r, cr, s_, b_, rhos, is_eps, act_idx, logprobs[b][act_idx], edge, terminal, is_accepts)

    def create_cycler_trajectory(self, traj):
        ltl_rewards = np.zeros(len(traj.rewards))
        cycle_rewards = np.array(traj.cycle_rewards)
        previous_visit_idx = 0
        #TODO: uncomment this and make it fast
        # for i in range(len(cycle_rewards)):
        #     if traj.accepts[i] > 0: # accepting state
        #         # get best cycle from previous_visit_idx to i
        #         cycle_idx = np.argmax(np.sum(cycle_rewards[previous_visit_idx:i + 1], axis=0))
        #         ltl_rewards[previous_visit_idx:i + 1] = cycle_rewards[previous_visit_idx:i + 1, cycle_idx]
        #         previous_visit_idx = i
        if self.baseline == "bhnr":
            # BHNR baseline case
            return self.create_bhnr_trajectory(traj)
        elif self.baseline == "tltl":
            return self.create_tltl_trajectory(traj)
        elif self.baseline == "quant":
            xformed_rewards = transform_qs_reward(cycle_rewards, self.lambda_val, self.min_rho_val)
            cycle_idx = np.argmax(np.sum(xformed_rewards[previous_visit_idx:len(xformed_rewards) + 1], axis=0))
        else:
            cycle_idx = np.argmax(np.sum(cycle_rewards[previous_visit_idx:len(cycle_rewards) + 1], axis=0))
        ltl_rewards[previous_visit_idx:len(cycle_rewards) + 1] = cycle_rewards[previous_visit_idx:len(cycle_rewards) + 1, cycle_idx]
        traj.ltl_rewards = list(ltl_rewards)
        assert(len(traj.ltl_rewards) == len(traj.rewards))
        return traj
    
    def create_bhnr_trajectory(self, traj):
        traj_rhos = np.array(traj.rhos)
        assert self.window_size is not None
        windows = self.get_stl_input(traj_rhos)
        qs_values = np.zeros(len(traj.rewards))
        current = 0
        # mtl_iterator = eval_mtl(self.parsed_formula, dt=1)
        for window in windows:
            bound = min(current + self.window_size, len(traj_rhos))
            rob_slice = argus.eval_robust_semantics(self.parsed_formula, window)
            for indx in range(current, bound):
                qs_values[indx] = (0 - rob_slice.at(indx)) / (0 - self.min_rho_val) # normalize between 0 and 1
            current += self.window_size
        traj.ltl_rewards = list(qs_values * self.lambda_val)
        assert(len(traj.ltl_rewards) == len(traj.rewards))
        return traj

    def create_tltl_trajectory(self, traj):
        traj_rhos = np.array(traj.rhos)
        stl_input = self.get_stl_input(traj_rhos)
        qs_values = np.zeros(len(traj.rewards))
        # normalize between 0 and 1
        qs_values[-1] = (0 - argus.eval_robust_semantics(self.parsed_formula, stl_input).at(0)) / (0 - self.min_rho_val)
        traj.ltl_rewards = list(qs_values * self.lambda_val)
        assert(len(traj.ltl_rewards) == len(traj.rewards))
        return traj

    def get_stl_input(self, traj_rhos):
        stl_input = {ap: [] for ap in self.rho_alphabet}
        for idx, rhos in enumerate(traj_rhos):
            for ridx in range(len(rhos)):
                stl_input[self.rho_alphabet[ridx]].append((idx, rhos[ridx]))
        if self.window_size is not None:
            windows = []
            for idx in range(0, len(traj_rhos), self.window_size):
                bound = min(idx + self.window_size + 1, len(traj_rhos))
                pretrace = {ap: argus.FloatSignal.from_samples(stl_input[ap][idx:bound]) for ap in self.rho_alphabet}
                windows.append(argus.Trace(pretrace))
            return windows
        else:
            for ap in stl_input:
                stl_input[ap] = argus.FloatSignal.from_samples(stl_input[ap])
            return argus.Trace(stl_input)

        
    def make_trajectories(self, env, s, b, a, r, cr, s_, b_, rhos, act_idx, is_eps, logprobs, edge, terminal):
        if not is_eps:
            assert act_idx == 0
            current_terminal_buchis = set([traj.get_last_buchi() for traj in self.trajectories if not traj.done])

            for buchi_state in range(env.observation_space['buchi'].n):
                if (self.main_trajectory is None) and (buchi_state == b): 
                    self.main_trajectory = len(self.trajectories)
                if buchi_state in current_terminal_buchis: continue
                traj = Trajectory(self.action_placeholder)
                next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, buchi_state)
                cr_, _ = env.compute_cycle_rewards(terminal, buchi_state, next_buchi_state, rhos)
                if accepting_rejecting_neutal < 1: 
                    traj.add(s, buchi_state, a, r, cr_, s_, next_buchi_state, rhos, is_eps, act_idx, \
                             logprobs[buchi_state][act_idx], edge, terminal, accepting_rejecting_neutal)
                    self.trajectories.append(traj)
            
                # also add epsilon transition 
                try:                        
                    for eps_idx in range(env.action_space[buchi_state].n):
                        # import pdb; pdb.set_trace()
                        traj = Trajectory(self.action_placeholder)
                        
                        # make epsilon transition
                        next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s, buchi_state, eps_idx)
                        cr_, _ = env.compute_cycle_rewards(terminal, buchi_state, next_buchi_state, rhos)
                        traj.add(s, buchi_state, a, r, cr_, s, next_buchi_state, rhos, True, 1 + eps_idx, \
                                 logprobs[buchi_state][1 + eps_idx], edge, terminal, accepting_rejecting_neutal)

                        # TODO: double check this part
                        # resync trajectory with s_
                        next_next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, next_buchi_state)
                        cr2, _ = env.compute_cycle_rewards(terminal, next_buchi_state, next_next_buchi_state, rhos)
                        traj.add(s, next_buchi_state, a, r, cr2, s_, next_next_buchi_state, rhos, \
                                 is_eps, act_idx, logprobs[next_buchi_state][act_idx], edge, terminal, accepting_rejecting_neutal)
                        self.trajectories.append(traj)
                except:
                    pass
        else:
            pass

    def update_trajectories(self, env, s, b, a, r, cr, s_, b_, rhos, act_idx, is_eps, logprobs, edge, terminal):
        new_trajectories = []
        if not is_eps:
            # update all trajectories
            for traj in self.trajectories:
                if traj.done == True: continue
                buchi_state = traj.get_last_buchi()

                # First add epsilon transitions if possible
                try:                        
                    for eps_idx in range(env.action_space[buchi_state].n):
                        # tic = time.time()
                        traj_copy = traj.copy()
                        # traj_copy = Trajectory(self.action_placeholder)
                        # print('CopyTime', time.time() - tic)
                        
                        # make epsilon transition
                        next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s, buchi_state, eps_idx)
                        cr_, _ = env.compute_cycle_rewards(terminal, buchi_state, next_buchi_state, rhos)
                        traj_copy.add(s, buchi_state, a, r, cr_, s, next_buchi_state, rhos, True, \
                                      1 + eps_idx, logprobs[buchi_state][1 + eps_idx], edge, terminal, accepting_rejecting_neutal)
                    
                        # resync trajectory with s_
                        next_next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, next_buchi_state)
                        cr2, _ = env.compute_cycle_rewards(terminal, next_buchi_state, next_next_buchi_state, rhos)
                        # TODO: double check this part
                        traj_copy.add(s, next_buchi_state, a, r, cr2, s_, next_next_buchi_state, rhos,\
                                      is_eps, act_idx, logprobs[next_buchi_state][act_idx], edge, terminal, accepting_rejecting_neutal)
                        new_trajectories.append(traj_copy)
                except:
                    pass

                next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, buchi_state)
                cr_, _ = env.compute_cycle_rewards(terminal, buchi_state, next_buchi_state, rhos)
                traj.add(s, buchi_state, a, r, cr_, s_, next_buchi_state, rhos, is_eps, act_idx, \
                         logprobs[buchi_state][act_idx], edge, terminal, accepting_rejecting_neutal)
                # figure it out here
        else:
            # only update main, non-hallucinated, trajectory.
            if len(self.trajectories) == 0: return
            traj = self.trajectories[self.main_trajectory]
            buchi_state = traj.get_last_buchi()
            next_buchi_state, accepting_rejecting_neutal = env.next_buchi(s_, buchi_state)
            cr_, _  = env.compute_cycle_rewards(terminal, buchi_state, next_buchi_state, rhos)
            traj.add(s, buchi_state, a, r, cr_, s_, next_buchi_state, rhos, is_eps, act_idx, \
                     logprobs[buchi_state][act_idx], edge, terminal, accepting_rejecting_neutal)
            # figure it out here
        for traj in new_trajectories:
            self.trajectories.append(traj)
    
    def get_torch_data(self, gamma, N=5, offpolicy=False):
        self.restart_traj()
        all_states = []
        all_buchis = []
        all_actions = []
        all_rewards = []
        all_ltl_rewards = []
        all_action_idxs = []
        all_logprobs = []
        all_next_buchis = []
        all_edges = []
        all_terminals = []

        if offpolicy:
            # form a batch of trajectories
            trajbatch = []
            if len(self.all_accept_trajectories) != 0:
                rewardtrajs = self.all_accept_trajectories
            else:
                rewardtrajs = self.all_reward_trajectories
            for X in [rewardtrajs, self.all_no_reward_trajectories]:
                try:
                    idxs = np.random.randint(0, len(X),size=N)
                except:
                    idxs = [] # len(self.all_reward_traj) == 0
                for idx in idxs:
                    traj = X[idx]
                    trajbatch.append(traj)
        else:
            trajbatch = self.batch_trajectories
        # all_dones = []
        # import pdb; pdb.set_trace()
        for traj in trajbatch:
            traj = self.create_cycler_trajectory(traj)
            rewards = []
            discounted_reward = 0
            ltl_rewards = []
            discounted_lreward = 0
            for reward in reversed(traj.rewards):
                # print(f"reward: {reward}, discounted_reward: {discounted_reward}, gamma: {gamma}")
                discounted_reward = reward + (gamma * discounted_reward)
                if self.baseline == "no_mdp" or self.baseline == "cycler_no_mdp":
                    rewards.insert(0, 0)
                else:
                    rewards.insert(0, discounted_reward)
            for lreward in reversed(traj.ltl_rewards):
                # print(f"reward: {reward}, discounted_reward: {discounted_reward}, gamma: {gamma}")
                if self.quant:
                    discounted_lreward = lreward + (gamma * discounted_lreward)
                else:
                    discounted_lreward = lreward + (discounted_lreward)
                ltl_rewards.insert(0, discounted_lreward)
            all_rewards += rewards # extend list
            all_ltl_rewards += ltl_rewards
            all_states += traj.states
            all_actions += traj.actions
            all_action_idxs += traj.act_idxs
            all_logprobs += traj.logprobs
            all_buchis += traj.buchis
            all_next_buchis += traj.next_buchis
            all_edges += traj.edges
            all_terminals += traj.terminals
                # all_dones += traj.dones
        if len(all_rewards) != len(all_ltl_rewards):
            import pdb; pdb.set_trace()
        all_states = torch.squeeze(torch.tensor(np.array(all_states))).detach().to(device).type(torch.float)
        all_buchis = torch.squeeze(torch.tensor(np.array(all_buchis))).detach().to(device).type(torch.int64).unsqueeze(1).unsqueeze(1)
        all_next_buchis = torch.squeeze(torch.tensor(np.array(all_next_buchis))).detach().to(device).type(torch.int64).unsqueeze(1).unsqueeze(1)
        all_action_idxs = torch.squeeze(torch.tensor(np.array(all_action_idxs))).detach().to(device).type(torch.int64)
        all_actions = torch.squeeze(torch.tensor(np.array(all_actions))).detach().to(device)
        all_logprobs = torch.squeeze(torch.tensor(all_logprobs)).detach().to(device)
        all_rewards = torch.tensor(np.array(all_rewards), dtype=torch.float32).to(device)
        all_ltl_rewards = torch.tensor(np.array(all_ltl_rewards), dtype=torch.float32).to(device)
        # all_edges = torch.tensor(np.array(all_edges), dtype=torch.float32).to(device)
        # edge is a new data structure, not torch tensor
        all_edges = np.array(all_edges) 
        all_terminals = torch.tensor(np.array(all_terminals), dtype=torch.bool).to(device) #TODO: check make the terminal to bool type
        return all_states, all_buchis, all_actions, all_next_buchis, all_rewards, all_ltl_rewards, all_action_idxs, all_logprobs, all_edges, all_terminals #, all_dones

    def restart_traj(self):
        #add the current trajectories to the batch's trajectories
        self.batch_trajectories.extend(self.trajectories)
        self.all_accept_trajectories += [traj for traj in self.trajectories if traj.has_accepting]
        self.all_reward_trajectories += [traj for traj in self.trajectories if traj.has_reward]
        self.all_no_reward_trajectories += [traj for traj in self.trajectories if not traj.has_reward]
        self.trajectories = []
        self.main_traj = Trajectory(self.action_placeholder)

    def clear(self):
        self.batch_trajectories = []
        self.trajectories = []
        self.all_reward_trajectories = self.all_reward_trajectories[-self.max_:]
        self.all_no_reward_trajectories = self.all_no_reward_trajectories[-self.max_:]
        self.all_accept_trajectories = self.all_accept_trajectories[-self.max_:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, param, has_continuous_action_space=True):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space        
        self.temp = action_std_init
        self.min_temp = param['ppo']['min_action_temp']
        self.var_denominator = param['ppo']['var_denominator']
        self.state_dim = state_dim
        if has_continuous_action_space:
            self.action_dim = action_dim['mdp'].shape[0]
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init, requires_grad=True).to(device)
        else:
            self.action_dim = action_dim['total']
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init, requires_grad=True).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim['mdp'].shape[0], 64),
                            nn.Tanh(), #relu for other experiments
                            nn.Linear(64, 64),
                            nn.Tanh(),
                        )
            # with torch.no_grad():
            #     policy_final_layer.weight *= .01
            self.mean_head = nn.Sequential(
                            nn.Linear(64, state_dim['buchi'].n * self.action_dim),
                            nn.Tanh()
                        )
            self.log_std_head = nn.Sequential(
                            nn.Linear(64, state_dim['buchi'].n * self.action_dim),
                            nn.Tanh()
                        )
            self.action_switch = nn.Linear(64, state_dim['buchi'].n * action_dim['total']) # for buchi epsilons

            with torch.no_grad():
                # bias towards no epsilons in the beginning
                self.action_switch.bias[::action_dim['total']] = 5.

                # bias towards left turn
                                        

            self.main_shp = (state_dim['buchi'].n, self.action_dim)
            self.shp = (state_dim['buchi'].n, action_dim['total'])
            
            # Mask actions not available
            self.mask = torch.ones((state_dim['buchi'].n, action_dim['total'])).type(torch.bool)
            if action_dim['total'] > 1:
                for buchi in range(state_dim['buchi'].n):
                    if buchi in action_dim:
                        eps = 1 + action_dim[buchi].n
                    else:
                        eps = 1
                    self.mask[buchi, eps:] = False
        else:
            raise NotImplemented

        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim['mdp'].shape[0], 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, state_dim['buchi'].n)
                    )
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.temp = new_action_std
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std, requires_grad=True).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, buchi_state):
        
        is_eps = False

        if self.has_continuous_action_space:
            body = self.actor(state)
            action_head = self.mean_head(body)
            action_mean = torch.reshape(action_head, self.main_shp)[buchi_state]



            # action_switch_head_all = torch.reshape(self.action_switch(body), self.shp)
            # masked_head_all = torch.masked.MaskedTensor(action_switch_head_all, self.mask)
            # probs_all = F.softmax(masked_head_all)
            # probs = probs_all.to_tensor(0)
            
            action_switch_head_all = torch.reshape(self.action_switch(body), self.shp)
            action_switch = action_switch_head_all#[buchi_state]
            mask = self.mask#[buchi_state]

            ### Bias against epsilon actions by multiplication by +10
            # action_switch[..., 0] *= torch.sign(action_switch[..., 0]) * 100
            
            # Fix
            probs = self.masked_softmax(action_switch, mask, -1)
            # probs = probs_all.squeeze()

            # action_switch_head = action_switch_head_all[buchi_state]
            # masked_head = action_switch_head[:self.mask[buchi_state].sum()]
            # probs = F.softmax(masked_head)
            try:
                action_or_eps = Categorical(probs[buchi_state])
            except:
                import pdb; pdb.set_trace()
            act_or_eps = action_or_eps.sample()

            if act_or_eps == 0:
                # else:
                if self.temp > self.min_temp:  # don't use the learned entropy here.
                    cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
                else:
                    action_log_std_head = self.log_std_head(body)
                    action_log_std = torch.reshape(action_log_std_head, self.main_shp)[buchi_state] / self.var_denominator
                    std = action_log_std.exp()
                    cov_mat = torch.diag_embed(std)
                dist = MultivariateNormal(action_mean, cov_mat)
                #samp_dist = Normal(action_mean, cov_mat)
                action = dist.rsample()

                clipped_action = torch.clip(action, -1, 1)
                action = clipped_action

                action_logprob = dist.log_prob(action) + action_or_eps.log_prob(act_or_eps)
            else:
                is_eps = True
                action = act_or_eps
                action_logprob = action_or_eps.log_prob(act_or_eps)
            
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_mean = action
            action_logprob = dist.log_prob(action)

        return action.detach(), action_mean.detach(), int(act_or_eps.detach()), is_eps, action_logprob.detach(), torch.log(probs).detach()
    
    def masked_softmax(self, vec, mask, dim=1, tol=1e-7):
        float_mask = mask.float().to(device)
        masked_vec = vec * float_mask
        max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
        exps = torch.exp(masked_vec-max_vec)
        masked_exps = exps * float_mask
        masked_exps += tol # make sure you dont get -inf when log
        masked_sums = masked_exps.sum(dim, keepdim=True)
        zeros=(masked_sums == 0)
        masked_sums += zeros.float()
        return masked_exps/masked_sums

    def evaluate(self, state, buchi, action, action_idxs):
        if self.has_continuous_action_space:

            body = self.actor(state)
            action_head = self.mean_head(body)
            action_means = torch.reshape(action_head, (-1,) + self.main_shp)
            action_mean = torch.take_along_dim(action_means, buchi, dim=1).squeeze()
            #import pdb; pdb.set_trace()
            action_log_std_head = self.log_std_head(body)
            action_log_stds = torch.reshape(action_log_std_head, (-1,) + self.main_shp)
            action_log_std = torch.take_along_dim(action_log_stds, buchi, dim=1).squeeze()

            action_switch_head_all = torch.reshape(self.action_switch(body), (-1,) + self.shp)
            action_switch = torch.take_along_dim(action_switch_head_all, buchi, dim=1)
            mask = torch.take_along_dim(self.mask.unsqueeze(0).repeat(len(action), 1, 1).to(device), buchi, dim=1)

            ### Bias against epsilon actions by multiplication by +10
            # action_switch[..., 0] *= torch.sign(action_switch[..., 0]) * 100
            
            ### Bug: Disturbs gradient flow
            # masked_switch = torch.masked.MaskedTensor(action_switch, mask)
            # probs_all = F.softmax(masked_switch, dim=-1)

            # Fix
            probs_all = self.masked_softmax(action_switch, mask, -1)
            probs = probs_all.squeeze()
            if len(probs.shape) == 1:  # if it's just 1d batch, unsqueeze it
                probs = probs.unsqueeze(1)
            dist_coinflip = Categorical(probs)
            

            action_var = self.action_var.expand_as(action_mean)
            action_std = action_log_std.exp()
            if self.temp > self.min_temp:
                cov_mat = torch.diag_embed(action_var).to(device)
                dist = MultivariateNormal(action_mean, cov_mat)
            else:
                cov_mat = torch.diag_embed(action_std).to(device)
                dist = MultivariateNormal(action_mean, cov_mat)
            dist_gaussian = dist.entropy()#.mean()
            #cov_mat = torch.diag_embed(action_var).to(device)

            
            action_logprobs = dist.log_prob(action)
            try:
                s_probs = torch.take_along_dim(probs, action_idxs.unsqueeze(1), dim=1).squeeze() + 1e-8
                logprobs_from_coinflip = torch.log(s_probs)
            except:
                logprobs_from_coinflip = torch.log(probs)

            # If controller take (a == 0) then LOG(P(a==0) * Normal(A))
            # If controller take (epsilon transition) then LOG(P(A))
            log_probs = logprobs_from_coinflip + action_logprobs * (action_idxs == 0)
            #log_probs = action_logprobs
            # State values 
            state_values = torch.take_along_dim(self.critic(state), buchi.squeeze(-1), dim=1).squeeze()

            # Entropy. Overapprox, not exact. RECHECK
            dist_coinflip = dist_coinflip.entropy().squeeze()
            #dist_gaussian = dist.entropy()#.mean()
            dist_entropy = dist_coinflip + dist_gaussian * probs[:, 0]
            #dist_entropy = dist_gaussian  #TODO: fix this!
            #import pdb; pdb.set_trace()

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_values = self.critic(state)
        return log_probs, state_values, dist_entropy
