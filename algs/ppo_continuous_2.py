import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from utls.utls import parse_stl_into_tree

import logger
import matplotlib.pyplot as plt
from utls.plotutils import plot_something_live
import numpy as np
from policies.actor_critic import RolloutBuffer, ActorCritic
import time
import wandb
from tqdm import tqdm
from PIL import Image


################################## set device ##################################
# print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")
# torch.default_device(device)

class PPO:
    def __init__(self, 
            env_space, 
            act_space, 
            gamma, 
            param,
            rho_val, 
            rho_alphabet,
            stl_phi,
            to_hallucinate=False,
            model_path=None
            ) -> None:

        lr_actor = param['ppo']['lr_actor']
        lr_critic = param['ppo']['lr_critic']
        self.K_epochs = param['ppo']['K_epochs']
        self.batch_size = param['ppo']['batch_size']
        self.eps_clip = param['ppo']['eps_clip']
        self.has_continuous_action_space = True
        action_std_init = param['ppo']['action_std']
        self.original_temp = param['ppo']['action_std']
        self.temp = param['ppo']['action_std']
        self.alpha = param['ppo']['alpha']
        self.ltl_lambda = param['lambda']
        self.original_lambda = param['lambda']
        self.qs_lambda = param['lambda_qs']
        self.off_policy_updates = param["ppo"]["off_policy"]
        self.rho_alphabet = rho_alphabet
        self.stl_phi = stl_phi

        self.policy = ActorCritic(env_space, act_space, action_std_init, param).to(device)
        if model_path and model_path != "":
            self.policy.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.mean_head.parameters(), 'lr': lr_actor},
                        {'params': self.policy.log_std_head.parameters(), 'lr': lr_actor},
                        {'params': self.policy.action_switch.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                    ])

        self.policy_old = ActorCritic(env_space, act_space, action_std_init, param).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        if param['baseline'] in ['quant', 'bhnr', 'tltl']:
            lambdaval = self.qs_lambda
        else:
            lambdaval = self.ltl_lambda
        if param['baseline'] == 'bhnr':
            stl_window = param['argus']['stl_window']
        else:
            stl_window = None
        self.buffer = RolloutBuffer(
            env_space['mdp'].shape, 
            act_space['mdp'].shape,
            lambdaval, 
            rho_val,
            self.rho_alphabet,
            self.stl_phi,
            param['baseline'],
            param['replay_buffer_size'], 
            to_hallucinate,
            stl_window=stl_window
            )
        
        self.gamma = gamma
        self.num_updates_called = 0
        self.MseLoss = nn.MSELoss()
    
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def select_action(self, state, is_testing):
        mdp_state = state['mdp']
        if isinstance(mdp_state, dict):
            mdp_state = mdp_state['state']
        with torch.no_grad():
            state_tensor = torch.FloatTensor(mdp_state).to(device)
            buchi = state['buchi']
            action, action_mean, action_idx, is_eps, action_logprob, all_logprobs = self.policy_old.act(state_tensor, buchi)
            if is_testing:
                if not is_eps:
                    return action_mean, action_idx, is_eps, action_logprob, all_logprobs
            return action, action_idx, is_eps, action_logprob, all_logprobs
        
    def save_model(self, save_path):
        torch.save(self.policy.state_dict(), save_path)
    
    def decay_temp(self, decay_rate, min_temp, decay_type):
        
        if decay_type == 'linear':
            self.temp = self.temp - decay_rate
        elif decay_type == 'exponential':
            self.temp = self.temp * decay_rate
        else:
            pass # in the learned entropy setting
        
        if (self.temp <= min_temp):
            self.temp = min_temp
        
        self.temp = round(self.temp, 4)
        #print(f'Setting temperature: {self.temp}')
        self.set_action_std(self.temp)

    def update(self):
        self.num_updates_called += 1

        # Optimize policy for K epochs
        for k in range(self.K_epochs):
            
            # TODO: update the RolloutBuffer to return rhos, edge, terminal
            old_states, old_buchis, old_actions, old_next_buchis, rewards, \
                lrewards, old_action_idxs, old_logprobs, old_edges, old_terminals \
                    = self.buffer.get_torch_data(
                        self.gamma, self.batch_size, offpolicy=self.off_policy_updates
                    )
            
            if len(old_states) == 0:
                # No signal available
                self.buffer.clear()
                return
            # use constrained optimization to modify the reward based on current lambda
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_buchis, old_actions, old_action_idxs)
                
            # optimize the combined reward
            crewards = lrewards + rewards
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = crewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_grad = -torch.min(surr1, surr2) 
            val_loss = self.MseLoss(state_values, crewards) 
            entropy_loss = dist_entropy
            # if (rewards == 0).all():
            #     # No signal available
            #     loss = 0.5*val_loss #- 0.01*entropy_loss 
            # else:
            # normalized_val_loss = val_loss
            if self.ltl_lambda != 0:
                normalized_val_loss = val_loss / (self.ltl_lambda) #/ (1 - self.gamma))
            else:
                normalized_val_loss = val_loss

            loss = policy_grad + 0.5*normalized_val_loss - self.alpha*entropy_loss #TODO: tune the amount we want to regularize entropy
            logger.logkv('policy_grad', policy_grad.detach().mean())
            logger.logkv('val_loss', val_loss.detach().mean())
            logger.logkv('entropy_loss', entropy_loss.detach().mean())
            logger.logkv('rewards', crewards.mean())
            # take gradient step

            self.optimizer.zero_grad()
            loss.mean().backward()
            # import pdb; pdb.set_trace()
            # torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=2.0, norm_type=2)
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        # if self.num_updates_called > 125 and self.num_updates_called % 25 == 0:
        #     import pdb; pdb.set_trace()
        self.buffer.clear()
        return loss.mean(), {"policy_grad": policy_grad.detach().mean(), "val_loss": normalized_val_loss.detach().item(), "entropy_loss": entropy_loss.detach().mean()}
    

def rollout(env, agent, param, i_episode, runner, testing=False, visualize=False, save_dir=None, eval=False):
    states, buchis = [], []
    state, _ = env.reset()
    states.append(state['mdp'])
    buchis.append(state['buchi'])
    initial_buchi = state['buchi']
    agent.buffer.restart_traj()
    mdp_ep_reward = 0
    constr_ep_reward = 0
    total_buchi_visits = 0
    # if not testing: 
    buchi_visits = []
    mdp_rewards = []
    ltl_rewards = []
    sum_xformed_rewards = np.zeros(env.num_cycles)
    
    # total_action_time = 0
    # total_experience_time = 0
    if eval:
        horizon = param['eval_horizon']
    else:
        horizon = param['ppo']['T']
    for t in range(1, horizon):  # Don't infinite loop while learning
        # tic = time.time()
        action, action_idx, is_eps, log_prob, all_logprobs = agent.select_action(state, testing)
        # total_action_time += time.time() - tic 

        if is_eps:
            action = int(action)
        else:
            action = action.cpu().numpy().flatten()
        
        # TODO: update the env_step function to return edge, terminal as info
        # try:
        next_state, mdp_reward, done, info = env.step(action, is_eps)
        if testing and visualize:
            if env.mdp.render_live:
                env.render()
        edge = info['edge']
        terminal = info['is_rejecting']
        ltl_r, _ = env.compute_cycle_rewards(terminal, state['buchi'], next_state['buchi'], info['rhos'])
        visit_buchi = next_state['buchi'] in env.automaton.automaton.accepting_states
        
        # here, transform the ltl reward so that it can be used properly in computation
        og_ltl_r = ltl_r
        # transforming delta


        agent.buffer.add_experience(
            env, 
            state['mdp'], 
            state['buchi'], 
            action, 
            mdp_reward,
            og_ltl_r, # transformed ltl quantitative semantics for purpose of cycler computation
            next_state['mdp'], 
            next_state['buchi'], 
            info['rhos'],
            action_idx, 
            is_eps, 
            all_logprobs,
            edge,
            terminal,
            visit_buchi
            )
        # total_experience_time += time.time() - tic

        # if visualize:
        #     env.render()
        # agent.buffer.atomics.append(info['signal'])
        mdp_ep_reward += mdp_reward
        sum_xformed_rewards += og_ltl_r
        ltl_rewards.append(og_ltl_r)
        constr_ep_reward += (agent.original_lambda * (visit_buchi) + mdp_reward)
        buchi_visits.append(visit_buchi)
        mdp_rewards.append(mdp_reward)
        total_buchi_visits += visit_buchi
        states.append(next_state['mdp'])
        buchis.append(next_state['buchi'])
        if done:
            break
        state = next_state

    if visualize and not env.mdp.render_live:
        if eval:
            save_dir = save_dir
        else:
            save_dir = save_dir + "/trajectory.png" if save_dir is not None else save_dir
        img = env.render(states=states, save_dir=save_dir)
    else:
        img = None
    # kind of a hack, but sum up from the first traj in the buffer, which should be the full trajectory with shaped reward.
    cycler_traj = agent.buffer.create_cycler_trajectory(agent.buffer.main_traj)
    ltl_ep_reward = sum(cycler_traj.ltl_rewards)
    # import pdb; pdb.set_trace()
    return mdp_ep_reward, ltl_ep_reward, constr_ep_reward, total_buchi_visits, img, np.array(buchi_visits), np.array(mdp_rewards)
        
def run_ppo_continuous_2(param, runner, env, to_hallucinate=False, visualize=False, save_dir=None, save_model=False, agent=None, n_traj=None):
    if agent is None:
        agent = PPO(
            env.observation_space, 
            env.action_space, 
            param['gamma'], 
            param,
            env.mdp.rho_min - env.mdp.rho_max, 
            env.mdp.rho_alphabet,
            env.parsed_stl,
            to_hallucinate,
            model_path=param['load_path'],
            )
    
    fixed_state, _ = env.reset()
    best_creward = -1 * float('inf')
    all_crewards = []
    all_bvisit_trajs = []
    all_mdpr_trajs = []
    all_test_bvisits = []
    all_test_mdprs = []
    all_test_crewards = []
    test_creward = 0
    if n_traj is None:
        n_traj = param['ppo']['n_traj'] + param['ppo']['n_pretrain_traj']
    for i_episode in tqdm(range(n_traj)):
        # TRAINING
        # Get trajectory
        # tic = time.time()
        mdp_ep_reward, ltl_ep_reward, creward, bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, runner, testing=False, save_dir=save_dir)
        if i_episode % param['ppo']['update_freq__n_episodes'] == 0 or i_episode == 1:
            # import pdb; pdb.set_trace()
            losstuple = agent.update()
            if losstuple is not None:
                current_loss, loss_info = losstuple
        # toc2 = time.time() - tic
        # print(toc, toc2)
        if i_episode % param['ppo']['temp_decay_freq__n_episodes'] == 0:
            agent.decay_temp(param['ppo']['temp_decay_rate'], param['ppo']['min_action_temp'], param['ppo']['temp_decay_type'])
            
        all_crewards.append(creward)
        all_bvisit_trajs.append(bvisit_traj)
        all_mdpr_trajs.append(mdp_traj)
        if i_episode % param['testing']['testing_freq__n_episodes'] == 0:
            test_avg_bvisits = 0
            test_avg_mdp_reward = 0
            test_avg_ltl_reward = 0
            test_avg_creward = 0
            for trial in range(param['testing']['num_rollouts']):
                mdp_test_reward, ltl_test_reward, test_creward, bvisits, img, test_bvisit_traj, test_mdp_traj = rollout(env, agent, param, i_episode, runner, testing=False, visualize=visualize, save_dir=save_dir) #param['n_traj']-100) ))
                test_avg_bvisits += bvisits
                test_avg_mdp_reward += mdp_test_reward
                test_avg_ltl_reward += ltl_test_reward
                test_avg_creward += test_creward
            test_avg_bvisits /= param['testing']['num_rollouts']
            test_avg_mdp_reward /= param['testing']['num_rollouts']
            test_avg_ltl_reward /= param['testing']['num_rollouts']
            test_avg_creward /= param['testing']['num_rollouts']
            test_creward = test_avg_creward
            if test_creward >= best_creward and save_model:
                best_creward = test_creward
                agent.save_model(save_dir + "/" + param["model_name"] + ".pt")
            all_test_bvisits.append(test_avg_bvisits)
            all_test_mdprs.append(test_avg_mdp_reward)
            all_test_crewards.append(test_creward)
            testing = True
        else:
            testing = False
        if visualize and testing and not env.mdp.render_live:
            if i_episode >= 1 and i_episode % 1 == 0:
                if img is None:
                    to_log = save_dir + "/trajectory.png" if save_dir is not None else save_dir
                else:
                    to_log = img
                runner.log({#'Iteration': i_episode,
                        'R_LTL': ltl_test_reward,
                        'R_MDP': mdp_test_reward,
                        'LossVal': current_loss,
                        "PolicyGradLoss": loss_info["policy_grad"],
                        "MSEValLoss": loss_info["val_loss"],
                        'ActionTemp': agent.temp,
                        'EntropyLoss': loss_info["entropy_loss"],
                        "Test_R_LTL": test_avg_ltl_reward,
                        "Test_R_MDP": test_avg_mdp_reward,
                        "Buchi_Visits": test_avg_bvisits,
                        "Dual Reward": test_avg_creward,
                        "testing": wandb.Image(to_log)
                        })
        else:
            if i_episode >= 1 and i_episode % 1 == 0:
                runner.log({#'Iteration': i_episode,
                        'R_LTL': ltl_ep_reward,
                        'R_MDP': mdp_ep_reward,
                        'LossVal': current_loss,
                        "PolicyGradLoss": loss_info["policy_grad"],
                        "MSEValLoss": loss_info["val_loss"],
                        'ActionTemp': agent.temp,
                        'EntropyLoss': loss_info["entropy_loss"],
                        "Test_R_LTL": test_avg_ltl_reward,
                        "Test_R_MDP": test_avg_mdp_reward,
                        "Buchi_Visits": test_avg_bvisits,
                        "Dual Reward": test_avg_creward,
                        })
    # load the best model
    agent.policy.load_state_dict(torch.load(save_dir + "/" + param["model_name"] + ".pt", map_location=torch.device(device)))
    return agent, all_crewards, all_bvisit_trajs, all_mdpr_trajs, all_test_crewards, all_test_bvisits, all_test_mdprs

def eval_agent(param, env, agent, visualize=False, save_dir=None):
    if agent is None:
        agent = PPO(
        env.observation_space, 
        env.action_space, 
        param['gamma'], 
        param, 
        env.mdp.rho_min - env.mdp.rho_max,
        env.mdp.rho_alphabet,
        env.parsed_stl,
        True,
        model_path=param['load_path'])
    fixed_state, _ = env.reset()
    crewards = []
    mdp_rewards = []
    avg_buchi_visits = []
    print("Beginning evaluation.")
    for i_episode in tqdm(range(param["num_eval_trajs"])):
        img_path = save_dir + "/eval_traj_{}.png".format(i_episode) if save_dir is not None else save_dir
        mdp_ep_reward, ltl_ep_reward, creward, bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, None, testing=False, visualize=visualize, save_dir=img_path, eval=True)
        mdp_rewards.append(mdp_ep_reward)
        avg_buchi_visits.append(bvisits)
        crewards.append(creward)
        if img is not None:
            im = Image.fromarray(img)
            if img_path is not None:
                im.save(img_path)
    mdp_test_reward, ltl_test_reward, test_creward, test_bvisits, img, bvisit_traj, mdp_traj = rollout(env, agent, param, i_episode, None, testing=True, visualize=visualize, eval=True)
    if img is not None:
        im = Image.fromarray(img)
        if save_dir is not None:
            img_path = save_dir + "/eval_traj_fixed.png"
            im.save(img_path)
    print("Buchi Visits and MDP Rewards for fixed (test) policy at Eval Time:")
    print("Buchi Visits:", test_bvisits)
    print("MDP Reward:", mdp_test_reward)
    print("")
    print("Average Buchi Visits and Average MDP Rewards for stochastic policy at Eval Time:")
    print("Buchi Visits:", sum(avg_buchi_visits) / len(avg_buchi_visits))
    print("MDP Reward:", sum(mdp_rewards) / len(mdp_rewards))
    return test_bvisits, mdp_test_reward, sum(avg_buchi_visits) / len(avg_buchi_visits), sum(mdp_rewards) / len(mdp_rewards), sum(crewards) / len(crewards)