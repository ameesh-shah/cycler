import hydra
from pathlib import Path
import wandb
import torch
import numpy as np
import os
import argus
from omegaconf import OmegaConf
from datetime import datetime
from envs.abstract_env import Simulator
from automaton import Automaton, AutomatonRunner
from algs.ppo_continuous_2 import run_ppo_continuous_2, eval_agent, PPO
import pickle as pkl
ROOT = Path(__file__).parent

@hydra.main(config_path=str(ROOT / "cfgs"))
def main(cfg):
    np.random.seed(cfg.init_seed)
    seeds = [np.random.randint(1e6) for _ in range(cfg.n_seeds)]
    np.random.seed(seeds[0])
    automaton = AutomatonRunner(Automaton(**cfg['ltl']))
    baseline = cfg["baseline"]
    if 'continuous' not in cfg['classes']:
        method = "q_learning"
    else:
        method = "ppo"
    for seed in seeds:
        # make logging dir for wandb to pull from, if necessary
        save_dir = os.path.join(os.getcwd(), 'experiments', cfg['run_name'] + "_" + cfg["baseline"] + "_" + '_seed' + str(seed) + '_lambda' + str(cfg['lambda']))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        results_dict = {}
        results_path = save_dir + '/results_dict_{}.pkl'.format(seed)
        torch.manual_seed(seeds[0]) # just the environment changes per seed in an experiment.
        np.random.seed(seeds)
        env = hydra.utils.instantiate(cfg.env)
        reward_sequence, buchi_traj_sequence, mdp_traj_sequence, test_reward_sequence, test_buchi_sequence, test_mdp_sequence, eval_results = run_baseline(cfg, env, automaton, save_dir, baseline, seed, method=method)
        results_dict["crewards"] = reward_sequence
        results_dict["btrajs"] = buchi_traj_sequence
        results_dict["mdptrajs"] = mdp_traj_sequence
        results_dict["test_creward_values"] = test_reward_sequence
        results_dict["test_b_visits"] = test_buchi_sequence
        results_dict["test_mdp_rewards"] = test_mdp_sequence
        results_dict["buchi_eval"], results_dict["mdp_eval"], results_dict["cr_eval"] = eval_results[2], eval_results[3], eval_results[4]
        results_dict["evaltime_test_buchi_visits"], results_dict["evaltime_test_mdp_reward"] = eval_results[0], eval_results[1]
        with open(results_path, 'wb') as f:
            pkl.dump(results_dict, f)
    print(cfg)

def run_baseline(cfg, env, automaton, save_dir, baseline_type, seed, method="ppo"):
    if baseline_type == "ours" or baseline_type == "cycler":
        reward_type = 2
        to_hallucinate = True
    elif baseline_type == "no_mdp": # LCER baseline method without mdp
        reward_type = 3
        to_hallucinate = True
    elif baseline_type == "baseline":  # LCER baseline method
        reward_type = 1
        to_hallucinate = True
    elif baseline_type == "ppo_only":  # baseline method
        reward_type = 1
        to_hallucinate = False
    elif baseline_type == "quant":  # use cycler with QS.
        reward_type = 0
        to_hallucinate = False
    elif baseline_type == "bhnr":
        reward_type = -1
        to_hallucinate = False
    elif baseline_type == "tltl":
        reward_type = -1
        to_hallucinate = False
    elif baseline_type == "eval": # evaluate an existing model
        assert cfg["load_path"] is not None
        assert cfg["load_path"] != ""
    else:
        print("BASELINE TYPE NOT FOUND!")
        import pdb; pdb.set_trace()
    if baseline_type == "bhnr":
        stl_formula = argus.parse_expr(cfg['argus']['bhnr_formula'])
    elif baseline_type == "tltl":
        stl_formula = argus.parse_expr(cfg['argus']['formula'])
    else:
        stl_formula = None  # won't be using this in reward computation
    train_trajs = cfg[method]['n_traj']
    run_name = cfg['run_name'] + "_" + baseline_type + "_" + '_seed' + str(seed) + '_lambda' + str(cfg['lambda']) + "_" + datetime.now().strftime("%m%d%y_%H%M%S")

    total_crewards = []
    total_buchis = []
    total_mdps = []
    total_test_crewards = []
    total_test_buchis = []
    total_test_mdps = []
    if baseline_type != "eval":
        with wandb.init(project="stlq", config=OmegaConf.to_container(cfg, resolve=True), name=run_name) as run:
            sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=reward_type, mdp_multiplier=cfg['mdp_multiplier'], stl_formula=stl_formula)
            agent, full_orig_crewards, buchi_trajs, mdp_trajs, all_test_crewards, all_test_bvisits, all_test_mdprs = run_ppo_continuous_2(cfg, run, sim, to_hallucinate=to_hallucinate, visualize=cfg["visualize"],
                                                            save_dir=save_dir, save_model=True, agent=None, n_traj=train_trajs)
            total_crewards.extend(full_orig_crewards)
            total_buchis.extend(buchi_trajs)
            total_mdps.extend(mdp_trajs)
            total_test_crewards.extend(all_test_crewards)
            total_test_buchis.extend(all_test_bvisits)
            total_test_mdps.extend(all_test_mdprs)
            if baseline_type == "ours" or baseline_type == "quant":
                traj_dir = save_dir + '/trajectories'
                if not os.path.exists(traj_dir):
                    os.mkdir(traj_dir)
            else:
                traj_dir = None
            run.finish()
    else:
        # in evaluation mode
        sim = Simulator(env, automaton, cfg['lambda'], qs_lambda=cfg['lambda_qs'], reward_type=0, mdp_multiplier=cfg['mdp_multiplier'])
        traj_dir = save_dir + '/trajectories'
        if not os.path.exists(traj_dir):
            os.mkdir(traj_dir)
        agent = None
        # define agent here and load the existing model path (need to import from policy files)

    test_bvisits, test_mdprew, buchi_visits, mdp_reward, combined_rewards = eval_agent(cfg, sim, agent, save_dir=traj_dir)
    return total_crewards, total_buchis, total_mdps, total_test_crewards, total_test_buchis, total_test_mdps, (test_bvisits, test_mdprew, buchi_visits, mdp_reward, combined_rewards)
    

if __name__ == "__main__":
    main()