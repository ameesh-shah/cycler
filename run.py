import hydra
from pathlib import Path
import wandb
import torch
import numpy as np
from omegaconf import OmegaConf
from envs.abstract_env import Simulator
from automaton import Automaton, AutomatonRunner
from algs.Q_stl import run_Q_STL
from algs.Q_value_iter import run_value_iter
ROOT = Path(__file__).parent

@hydra.main(config_path=str(ROOT / "cfgs"))
def main(cfg):
    np.random.seed(cfg.init_seed)
    seeds = [np.random.randint(1e6) for _ in range(cfg.n_seeds)]
    torch.manual_seed(seeds[0])
    np.random.seed(seeds[0])
    env = hydra.utils.instantiate(cfg.env)
    automaton = AutomatonRunner(Automaton(**cfg['ltl']))
    sim = Simulator(env, automaton)
    with wandb.init(project="stlq", config=OmegaConf.to_container(cfg, resolve=True)) as run:
        #run_Q_STL(cfg, run, sim)
        run_value_iter(cfg, run, sim)
    print(cfg)


if __name__ == "__main__":
    main()










# import argparse
# import os
# import logger
# import yaml
# import torch
# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper
# from experiment_tools.factory import setup_params

# # from algs.ppo_ltl import run_ppo_ltl
# from algs.Q_learning import run_Q_learning
# from algs.Q_stl import run_Q_STL
# from algs.ppo import run_PPO
# import numpy as np
# from envs.abstract_env import Simulator
# #from automaton import Automaton, AutomatonRunner

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torch.masked.maskedtensor")

# # def main_stl(seed, param):
# #     # parse the given formula into an STLNode class
# def main(seed, param, to_redo):
#     # automaton = AutomatonRunner(Automaton(**param['ltl']))
#     # logger.info('*'*20 + '\tLTL: %s' % automaton.automaton.formula)
#     directory = os.path.join(param['logger']['dir_name'], 'stl_q', 'experiment_%05.f' % (seed) )
#     if ('discrete' in param['classes']) and (to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', directory))):
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         param['env']['file'] = param['classes']['discrete']
#         env = setup_params(param)

#         # Simple check to see if observation_space space is discrete
#         is_discrete_obs_space = True
        
#         logger.configure(name=directory)
#         #run_Q_learning(param, sim, False, not is_discrete_obs_space, to_hallucinate=True)
#         run_Q_STL(param, env)
        
#     # dir = os.path.join(param['logger']['dir_name'], 'baseline_q', 'experiment_%05.f' % (seed) )
#     # if ('discrete' in param['classes']) and (to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', dir))):
#     #     torch.manual_seed(seed)
#     #     np.random.seed(seed)
#     #     param['env']['file'] = param['classes']['discrete']
#     #     env = setup_params(param)
#     #     sim = Simulator(env, automaton)

#     #     # Simple check to see if observation_space space is discrete
#     #     is_discrete_obs_space = False
#     #     try:
#     #         sim.observation_space['mdp'].n
#     #         is_discrete_obs_space = True
#     #     except:
#     #         pass
        
#     #     logger.configure(name=dir)
#     #     run_Q_learning(param, sim, False, not is_discrete_obs_space, to_hallucinate=False)

#     # dir = os.path.join(param['logger']['dir_name'], 'ours_ppo', 'experiment_%05.f' % (seed) )
#     # if ('continuous' in param['classes']) and (to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', dir))):
#     #     torch.manual_seed(seed)
#     #     np.random.seed(seed)
#     #     param['env']['file'] = param['classes']['continuous']
#     #     env = setup_params(param)
#     #     sim = Simulator(env, automaton)

#     #     # Simple check to see if observation_space space is discrete
#     #     is_discrete_obs_space = False
#     #     try:
#     #         sim.observation_space['mdp'].n
#     #         is_discrete_obs_space = True
#     #         # PPO for discrete state space: NOT IMPLEMENTED
#     #         return
#     #     except:
#     #         pass
        
#     #     logger.configure(name=dir)
#     #     run_PPO(param, sim, False, not is_discrete_obs_space, to_hallucinate=True)
    
#     # dir = os.path.join(param['logger']['dir_name'], 'baseline_ppo', 'experiment_%05.f' % (seed) )
#     # if ('continuous' in param['classes']) and (to_redo or not os.path.exists(os.path.join(os.getcwd(), 'experiments', dir))):
#     #     torch.manual_seed(seed)
#     #     np.random.seed(seed)
#     #     param['env']['file'] = param['classes']['continuous']
#     #     env = setup_params(param)
#     #     sim = Simulator(env, automaton)

#     #     # Simple check to see if observation_space space is discrete
#     #     is_discrete_obs_space = False
#     #     try:
#     #         sim.observation_space['mdp'].n
#     #         is_discrete_obs_space = True
#     #         # PPO for discrete state space: NOT IMPLEMENTED
#     #         return
#     #     except:
#     #         pass
        
#     #     logger.configure(name=dir)
#     #     run_PPO(param, sim, False, not is_discrete_obs_space, to_hallucinate=False)

# if __name__ == '__main__':
#     # Local:
#     # python run.py chain.yaml

#     parser = argparse.ArgumentParser(description='Run Experiment')

#     parser.add_argument('cfg', help='config file', type=str)
#     parser.add_argument('-r', '--restart', action='store_true')
#     args = parser.parse_args()

#     assert args.cfg.endswith('.yaml'), 'Must be yaml file'
#     with open(os.getcwd() + '/cfgs/{0}'.format(args.cfg), 'r') as f:
#         param = yaml.load(f, Loader=Loader)
    
    
#     np.random.seed(param['init_seed'])
#     seeds = [np.random.randint(1e6) for _ in range(param['n_seeds'])]

#     for seed in seeds:
#         print('*' * 20)
#         # param['logger']['name'] = #'experts_and_'+param['MCTS']['bandit_strategy'] if param['experiment']['experts'] else 'policy_and_'+param['MCTS']['bandit_strategy']
#         # logger.configure(name=os.path.join(param['logger']['dir_name'], 'experiment_%05.f' % (seed) ))
#         logger.Logger.set_level(logger,logger.DEBUG)
#         logger.info("Seed = {}".format(float(seed)))
#         main(seed, param, args.restart)