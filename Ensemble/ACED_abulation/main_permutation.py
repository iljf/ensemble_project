# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle
import wandb
import pandas as pd

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import ensemble_test
from util_wrapper import *
import torch.nn.functional as F




def global_seed_initailizer(seed):
    # random seed for numpy
    np.random.seed(seed)
    # random seed for torch
    torch.manual_seed(seed)
    # random seed for torch.cuda
    torch.cuda.manual_seed(seed)

def predefined_scheduler(schedule_mode=1, env_name = 'road_runner', action_prob_set = None, min_max_action_prob = [0.1, 0.9], debug = False):
        if env_name == 'road_runner':
            # there are two rewarding modes: 0: default, 1: kill koyote
            reward_mode_info = {0: 'default', 1: 'kill_koyote'}
        elif env_name == 'frostbite':
            # there are two rewarding modes: 0: default, 1: collect_fish
            reward_mode_info = {0: 'default', 1: 'jumb_forever'}
        elif env_name == 'crazy_climber':
            # there are two rewarding modes: 0: default, 1: get_hit
            reward_mode_info = {0: 'default', 1: 'get_hit'}
        elif env_name == 'jamesbond':
            # there are two rewarding modes: 0: default, 1:
            reward_mode_info = {0: 'default', 1: 'dodge_everything'}
        elif env_name == 'kangaroo':
            reward_mode_info = {0: 'default', 1: 'punch_monkeys'}
        elif env_name == 'chopper_command':
            reward_mode_info = {0: 'default', 1: 'ignore jets'}
        elif env_name == 'bank_heist':
            reward_mode_info = {0: 'default', 1: 'car persuit'}
        elif env_name == 'assault':
            reward_mode_info = {0: 'default', 1: 'car persuit'}
        elif env_name == 'krull':
            reward_mode_info = {0: 'default', 1: 'car persuit'}
        elif env_name == 'alien':
            reward_mode_info = {0: 'default', 1: 'track aliens'}
        elif env_name == 'asterix':
            reward_mode_info = {0: 'default', 1: 'objective reward decline'}
        elif env_name == 'battle_zone':
            reward_mode_info = {0: 'default', 1: 'focus on airships'}
        elif env_name == 'boxing':
            reward_mode_info = {0: 'default', 1: '1 for punch 2 for jab'}
        elif env_name == 'amidar':
            reward_mode_info = {0: 'default', 1: 'walking through maze'}
        elif env_name == 'breakout':
            reward_mode_info = {0: 'default', 1: 'reverse reward'}
        elif env_name == 'demon_attack':
            reward_mode_info = {0: 'default', 1: 'Shoot demons'}
        elif env_name == 'freeway':
            reward_mode_info = {0: 'default', 1: 'negative reward for mistakes'}
        elif env_name == 'gopher':
            reward_mode_info = {0: 'default', 1: 'kill gopher'}
        elif env_name == 'hero':
            reward_mode_info = {0: 'default', 1: 'use all dynamites'}
        elif env_name == 'kung_fu_master':
            reward_mode_info = {0: 'default', 1: 'use more kicks'}
        elif env_name == 'ms_pacman':
            reward_mode_info = {0: 'default', 1: 'collect fruits and ghosts'}
        elif env_name == 'pong':
            reward_mode_info = {0: 'default', 1: 'half the point for player'}
        elif env_name == 'private_eye':
            reward_mode_info = {0: 'default', 1: 'additional points for getting hit by a brick or flowerpot'}
        elif env_name == 'qbert':
            reward_mode_info = {0: 'default', 1: 'kill snakes'}
        elif env_name == 'seaquest':
            reward_mode_info = {0: 'default', 1: 'kill fish'}
        elif env_name == 'up_n_down':
            reward_mode_info = {0: 'default', 1: 'negative reward for moving'}


        # if action_prob_set is None:
        #     action_prob_set = np.random.rand(4) * (min_max_action_prob[1] - min_max_action_prob[0]) + min_max_action_prob[0]
        # last iterations to be 0
        if action_prob_set is None:
            action_prob_set = np.random.rand(3) * (min_max_action_prob[1] - min_max_action_prob[0]) + min_max_action_prob[0]

        else:
            if len(action_prob_set) != 4:
                raise ValueError('action_prob_set should be of length 4')


        """
        reward mode sampling 할때 마지막 400k 에서 500k를 0으로 세팅 할때
        rand_cond_seed = np.append(rand_cond_seed, 0) 으로 뒤에 0을 하나더 넣어줌   
        """

        ## reward mode schedule
        # mix the predefined reward modes
        rand_cond_seed = [[j for _ in range((5-1)//len(reward_mode_info.keys()))] for j in range(len(reward_mode_info.keys()))]
        rand_cond_seed = np.array(rand_cond_seed).flatten()

        # random shuffle of the predefined reward modes
        np.random.shuffle(rand_cond_seed)
        rand_cond_seed = np.append(0, rand_cond_seed)
        # repeat each of them 100k times
        reward_mode_schedule = np.repeat(rand_cond_seed, 200000)
        # overide the last block to be 1
        reward_mode_schedule[-200000:] = 1

        # TODO check the code;
        # repeat by 100k times till 400k
        # num_repeats= 4
        # reward_mode_seed = (rand_cond_seed * num_repeats)
        # reward_mode_schedule = np.repeat(reward_mode_seed, 100000)

        ## action probability schedule # continuous / discrete
        if schedule_mode % 2 == 0: # if schedule_mode is 0 ,2,4 ,6 then discrete
            action_prob_seed = np.array(action_prob_set)
            # random shuffle of the predefined reward modes
            np.random.shuffle(action_prob_seed)
            action_prob_seed = np.append(0, action_prob_seed)
            action_prob_seed = np.round(action_prob_seed, 1)  # round to 1 decimal (e.g. 0.1, 0.5)
            # last 100k to be 0
            action_prob_seed = np.append(action_prob_seed, 0)
            # repeat each of them 100k times

            action_prob_seed_schedule = np.repeat(action_prob_seed, 200000)

            # TODO check the code;
            # repeat by 100k times till 400k
            # action_mode_seed = (action_prob_seed * num_repeats)
            # action_prob_seed_schedule = np.repeat(action_mode_seed, 100000)

        else: # if schedule_mode is 1,3,5,7 then continuous
            action_prob_seed_schedule = np.random.rand(200000)/5 # TODO
            # or sine wave - alike + np.random.randn(500000)/10

        return reward_mode_schedule, action_prob_seed_schedule, reward_mode_info


if __name__ == '__main__':


    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--id', type=str, default='ACED_infer_AC', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=122, help='Random seed')
    parser.add_argument('--iteration', type=int, default=2, help='Number of iteration')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--game', type=str, default='assault', choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument('--T-max', type=int, default=int(1e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--gamma', type=float, default=0.6, metavar='-', help='Probability to sustain previous reliability score')
    parser.add_argument('--target-update', type=int, default=int(32000), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
    parser.add_argument('--learn-start', type=int, default=int(80000), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=5000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
    # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--memory', help='Path to save/load the memory from')
    parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
    # ensemble
    parser.add_argument('--num-ensemble', type=int, default=5, metavar='N', help='Number of ensembles')
    parser.add_argument('--beta-mean', type=float, default=1, help='mean of bernoulli')
    parser.add_argument('--temperature', type=float, default=40, help='temperature for CF')
    parser.add_argument('--ucb-infer', type=float, default=0, help='coeff for UCB infer')
    parser.add_argument('--ucb-train', type=float, default=10, help='coeff for UCB train')
    parser.add_argument('--mse_weights', type=float, default=0.8, metavar='-', help='reliability weights')
    parser.add_argument('--mse_temperature', type=float, default=0.3, help='temperature for mse_loss')
    parser.add_argument('--energy_temperature', type=float, default=0.8, help='temperature for Energy')
    parser.add_argument('--scheduler-mode', type=int, default=2, metavar='S', help='Scheduler seed/mode')
    parser.add_argument('--action-prob-max', type=float, default=0.5, help='max action probability')
    parser.add_argument('--action-prob-min', type=float, default=0.1, help='min action probability')
    parser.add_argument('--block-id', type=int, default=0, help='testing schedule block')
    parser.add_argument('--permutation', type=int, default=0, help='0 for original, 1 for average, 2 for random, 3 for mse')

    # Setup
    args = parser.parse_args()

    # wandb intialize
    # if args.id == 'ACED':
    #     wandb.init(project="block_rb",
    #                name="ACED_" + args.game + " " + "Seed" + str(args.seed) + "_i_" + str(args.iteration),
    #                config=args.__dict__
    #                )
    # if args.id == 'ACED_infer_AC':
    #     wandb.init(project="ACED_R_mse",
    #                name="ACED_" + args.game,
    #                config=args.__dict__
    #                )

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    # exp name
    exp_name = args.id + '/' + args.game
    exp_name += '/Beta_' + str(args.beta_mean) + '_T_' + str(args.temperature)
    exp_name +='_UCB_I_' + str(args.ucb_infer) + '_UCB_T_' + str(args.ucb_train) + '/'
    exp_name += '/seed_' + str(args.seed) + '/'
    exp_name += '/iteration_' + str(args.iteration) + '/'

    results_dir = os.path.join('./results', exp_name)
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)

    args.evaluate = True

    metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda')
        torch.cuda.manual_seed(np.random.randint(1, 10000))
        torch.backends.cudnn.enabled = args.enable_cudnn
    else:
        args.device = torch.device('cpu')

    # Simple ISO 8601 timestamped logger
    def log(s):
        print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)

    def load_memory(memory_path, disable_bzip):
        if disable_bzip:
            with open(memory_path, 'rb') as pickle_file:
                return pickle.load(pickle_file)
        else:
            with bz2.open(memory_path, 'rb') as zipped_pickle_file:

                return pickle.load(zipped_pickle_file)


    def memory_agent_sampling(mse_log, block_id, temperature=0.8):
        sampled_agents = []
        reliabilities = []
        block_ids = []

        for mse_vec in mse_log:
            mse_tensor = torch.tensor(mse_vec)
            reli = torch.softmax(-mse_tensor / temperature, dim=0).numpy()
            sampled_agent = np.random.choice(len(reli), p=reli)
            sampled_agents.append(sampled_agent)
            reliabilities.append(reli.tolist())
            block_ids.append(block_id)

        agent_sample_counts = {i: sampled_agents.count(i) for i in range(len(reli))}

        df = pd.DataFrame({
            'block_id': block_ids,
            'sampled_agent': sampled_agents,
            'reliability': reliabilities,
            'mse': mse_log
        })

        return df, agent_sample_counts


    def save_memory(memory, memory_path, disable_bzip):
        if disable_bzip:
            with open(memory_path, 'wb') as pickle_file:
                pickle.dump(memory, pickle_file)
        else:
            with bz2.open(memory_path, 'wb') as zipped_pickle_file:
                pickle.dump(memory, zipped_pickle_file)

    # Environment
    env = Env(args)
    env = Rewardvalue(env)
    env = Action_random(env, eps=0.1)
    env.train()
    action_space = env.action_space()

    # Agent
    dqn_list = []

    #TODO: Diverse models
    # args.num_ensemble 수 만큼 agent를 생성하고 i % len(models) 만큼 할당
    # Each agent with diff models
    # agnet idx 0: DQN, 1: DDQN, 2: NoisyDQN, 3: DuelingDQN, 4: DistributionalDQN
    models = ['DQN', 'DDQN', 'NoisyDQN', 'DuelingDQN', 'DistributionalDQN']
    for i in range(args.num_ensemble):
        model = models[i % len(models)]
        dqn = Agent(args, env, model, index=i) ## shared replay memory
        # dqn = Agent(args, env, model, ReplayMemory(args, args.memory_capacity, args.beta_mean, args.num_ensemble)) ## for individual replay memory
        dqn_list.append(dqn)

    # If a model is provided, and evaluate is false, presumably we want to resume, so try to load memory
    if args.model is not None and not args.evaluate:
        if not args.memory:
            raise ValueError('Cannot resume training without memory save path. Aborting...')
        elif not os.path.exists(args.memory):
            raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))
        mem = load_memory(args.memory, args.disable_bzip_memory)
    else:
        mem = ReplayMemory(args, args.memory_capacity, args.beta_mean, args.num_ensemble)

    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

    # scheduler
    global_seed_initailizer(args.seed)
    reward_mode_, action_probs_, info = predefined_scheduler(args.scheduler_mode, args.game, min_max_action_prob = [args.action_prob_min, args.action_prob_max])
    if args.block_id < 5:
        block_id = args.block_id # 0 = 0~100k, 1 = 100k~200k, 2 = 200k~300k
        reward_mode_, action_probs_ = reward_mode_[(block_id)*int(2e5):(block_id+1)*int(2e5)], action_probs_[(block_id)*int(2e5):(block_id+1)*int(2e5)]
        # reward_mode_, action_probs_, info = predefined_scheduler(args.scheduler_mode, args.game, min_max_action_prob = [args.action_prob_min, args.action_prob_max], debug=True)

    # Construct validation memory
    val_mem = ReplayMemory(args, args.evaluation_size, args.beta_mean, args.num_ensemble)
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            state, done = env.reset(), False
        next_state, _, done = env.step(np.random.randint(0, action_space))
        val_mem.append(state, None, None, done, torch.zeros(5, dtype=torch.float32))
        state = next_state
        T += 1

    all_sampled_results = []
    reward_log = []
    # Make data frame with evaluating
    if args.evaluate:
        for block_id in range(5):
            args.block_id = block_id

            global_seed_initailizer(args.seed)

            for en_index in range(args.num_ensemble):
                dqn_list[en_index].load_weights(block_id)

            reward_mode_, action_probs_, info = predefined_scheduler(args.scheduler_mode, args.game,
                                                                     min_max_action_prob=[args.action_prob_min,
                                                                                          args.action_prob_max])
            reward_mode_, action_probs_ = reward_mode_[(block_id) * int(2e5):(block_id + 1) * int(2e5)], action_probs_[(block_id) * int(2e5):(block_id + 1) * int(2e5)]

            env.eps = action_probs_[0]
            env.env.reward_mode = reward_mode_[0]
            action_p = env.eps
            scheduler = env.env.reward_mode


            for en_index in range(args.num_ensemble):
                dqn_list[en_index].eval()



        # KM: test code
            if args.permutation == 3:
                avg_reward, avg_Q, episode_reward, mse_log = ensemble_test(args, 0, dqn_list, val_mem, metrics, results_dir,
                                                                  num_ensemble=args.num_ensemble, scheduler=scheduler,
                                                                  action_p=action_p, evaluate=True, memory=mem)  # Test
                print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                df_sampled, counts = memory_agent_sampling(mse_log, block_id, temperature=args.energy_temperature)
                all_sampled_results.append(df_sampled)
            else:
                avg_reward, avg_Q, episode_reward, ep_reliability, T_mse = ensemble_test(args, 0, dqn_list, val_mem, metrics, results_dir,
                                                  num_ensemble=args.num_ensemble, scheduler=scheduler, action_p=action_p, evaluate=True, memory=mem)  # Test
                print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))

                for ep_idx, r in enumerate(episode_reward):
                    reward_log.append({
                        "block_id": block_id,
                        "episode": ep_idx,
                        "reward": r,
                        "avg_reward": avg_reward,
                        "reliability" : ep_reliability[ep_idx],
                        "mse" : T_mse[ep_idx]
                    })

        directory = args.id + '/' + args.game
        results_dir_ = os.path.join('./results', directory)
        os.makedirs(results_dir_, exist_ok=True)
        if args.permutation == 3:
            df_mse = pd.concat(all_sampled_results, ignore_index =True)
            df_mse.to_csv(os.path.join(results_dir_, f'{args.id}_{args.game}_mse.csv'))
            print("Done")
        else:
            df = pd.DataFrame(reward_log)
            df.to_csv(os.path.join(results_dir_, f'{args.id}_{args.game}.csv'))
            print("Done")

    else:
        # Training loop
        for en_index in range(args.num_ensemble):
            dqn_list[en_index].train()
        T, done = 0, True
        selected_en_index = np.random.randint(args.num_ensemble)

        reliability = torch.ones(args.num_ensemble, device=args.device) / args.num_ensemble
        while T < args.learn_start:
            if T % 10000 ==0:
                print(f"Running memory: {T}/{args.learn_start} for game: {args.game}")
            env.eps = action_probs_[T]
            env.env.reward_mode = reward_mode_[T]
            action_p = env.eps
            scheduler = env.env.reward_mode

            if done:
                state, done = env.reset(), False

            if T % args.replay_frequency == 0:
                dqn.reset_noise()

            # arbitration control
            if args.ucb_infer == 0:
                online_Qlist = []
                target_Qlist = []
                action_Qlist = []
                Q_list = []
                for en_index in range(args.num_ensemble):
                    online_Q, action = dqn_list[en_index].act_v2(state)
                    Q_list.append(online_Q)
                    action_Qlist.append(action)

                Q_tot = sum(Q_list[i] * reliability[i] for i in range(len(Q_list)))
                action = Q_tot.argmax().item()

                next_state, reward, done = env.step(action)
                nonterminal = 1.0 if not done else 0.0

                mse_list = []
                for en_index in range(args.num_ensemble):
                    online_Q = Q_list[en_index][0, action]
                    target_Q = dqn_list[en_index].get_target_q_mse(next_state)
                    target_Q = reward + (nonterminal * args.discount * target_Q)
                    target_Qlist.append(target_Q)

                    target_Q= torch.flatten(target_Q)
                    online_Q = torch.flatten(online_Q)

                    mse_loss = F.mse_loss(online_Q, target_Q)
                    mse_list.append(mse_loss)

                mse_list[-1] = args.mse_weights * mse_list[-1] # distributional dqn
                mse_tensor = torch.stack(mse_list)

                Energy = mse_tensor

                softmax_reliability = F.softmax(-mse_tensor / args.mse_temperature, dim=0)
                momentum_reliability = (1-args.gamma) * reliability + args.gamma * softmax_reliability
                momentum_reliability = torch.clamp(momentum_reliability, min=0.2, max=0.5)

                reliability = momentum_reliability / momentum_reliability.sum()

                state = next_state

                # UCB exploration
            if args.ucb_infer > 0:
                mean_Q, var_Q = None, None
                L_target_Q = []
                for en_index in range(args.num_ensemble):
                    target_Q = dqn_list[en_index].get_online_q(state)
                    L_target_Q.append(target_Q)
                    if en_index == 0:
                        mean_Q = target_Q / args.num_ensemble
                    else:
                        mean_Q += target_Q / args.num_ensemble
                temp_count = 0
                for target_Q in L_target_Q:
                    if temp_count == 0:
                        var_Q = (target_Q - mean_Q) ** 2
                    else:
                        var_Q += (target_Q - mean_Q) ** 2
                    temp_count += 1
                var_Q = var_Q / temp_count
                std_Q = torch.sqrt(var_Q).detach()
                ucb_score = mean_Q + args.ucb_infer * std_Q
                action = ucb_score.argmax(1)[0].item()

            # Choose an action greedily
            if args.ucb_infer == -1:
                action = dqn_list[selected_en_index].act(state)


            next_state, reward, done = env.step(action)

            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)
            mem.append(state, action, reward, done, Energy)
            state = next_state
            T += 1

        log('Memory:' + ' T = ' + str(T) + ' / ' + str(args.learn_start))

        # log q values for agent in action
        reliability = torch.ones(args.num_ensemble, device=args.device) / args.num_ensemble
        # Set reward mode, action prob according to the schedule
        for T in trange(1, args.T_max + 1):
            env.eps = action_probs_[T-1]
            env.env.reward_mode = reward_mode_[T-1]
            action_p = env.eps
            scheduler = env.env.reward_mode

            if done:
                state, done = env.reset(), False
                # selected_en_index = np.random.randint(args.num_ensemble)

            if T % args.replay_frequency == 0:
                dqn.reset_noise()

            if args.ucb_infer == 0:
                online_Qlist = []
                target_Qlist = []
                action_Qlist = []
                Q_list = []
                for en_index in range(args.num_ensemble):
                    online_Q, action = dqn_list[en_index].act_v2(state)
                    Q_list.append(online_Q)
                    action_Qlist.append(action)


                Q_tot = sum(Q_list[i] * reliability[i] for i in range(len(Q_list)))
                action = Q_tot.argmax().item()

                next_state, reward, done = env.step(action)
                nonterminal = 1.0 if not done else 0.0

                mse_list = []
                for en_index in range(args.num_ensemble):
                    online_Q = Q_list[en_index][0, action]
                    target_Q = dqn_list[en_index].get_target_q_mse(next_state)
                    target_Q = reward + (nonterminal * args.discount * target_Q)
                    target_Qlist.append(target_Q)

                    target_Q= torch.flatten(target_Q)
                    online_Q = torch.flatten(online_Q)

                    mse_loss = F.mse_loss(online_Q, target_Q)
                    mse_list.append(mse_loss)

                mse_list[-1] = args.mse_weights * mse_list[-1] # distributional dqn
                mse_tensor = torch.stack(mse_list)

                Energy = mse_tensor

                softmax_reliability = F.softmax(-mse_tensor / args.mse_temperature, dim=0)
                momentum_reliability = (1-args.gamma) * reliability + args.gamma * softmax_reliability
                momentum_reliability = torch.clamp(momentum_reliability, min=0.2, max=0.5)

                reliability = momentum_reliability / momentum_reliability.sum()

                state = next_state

            # UCB exploration
            if args.ucb_infer > 0:
                mean_Q, var_Q = None, None
                L_target_Q = []
                for en_index in range(args.num_ensemble):
                    target_Q = dqn_list[en_index].get_online_q(state)
                    L_target_Q.append(target_Q)
                    if en_index == 0:
                        mean_Q = target_Q / args.num_ensemble
                    else:
                        mean_Q += target_Q / args.num_ensemble
                temp_count = 0
                for target_Q in L_target_Q:
                    if temp_count == 0:
                        var_Q = (target_Q - mean_Q)**2
                    else:
                        var_Q += (target_Q - mean_Q)**2
                    temp_count += 1
                var_Q = var_Q / temp_count
                std_Q = torch.sqrt(var_Q).detach()
                ucb_score = mean_Q + args.ucb_infer * std_Q
                action = ucb_score.argmax(1)[0].item()
            if args.ucb_infer == -1:
                action = dqn_list[selected_en_index].act(state)  # Choose an action greedily (with noisy weights)

            next_state, reward, done = env.step(action)  # Step
            # scheduler.update(T)

            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            mem.append(state, action, reward, done, Energy)  # Append transition to memory


            # Train and test
            # if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
            if T % args.replay_frequency == 0:
                total_q_loss = 0

                q_loss_tot = 0
                idx_tot = []

                exps = mem.sample(args.batch_size, args.energy_temperature)
                for en_index, exp in enumerate(exps):
                    probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals, masks, weights = exp
                    idx_tot.append(tree_idxs)
                    q_loss, batch_loss, CE_loss, transition_loss = dqn_list[en_index].diversity_learn(idxs, states, actions, returns,
                                                               next_states, nonterminals, masks[:, en_index],
                                                               weights, None)

                    # Update loss of sampled transitions
                    for i, idx in enumerate(idxs):
                        current_mse = mem.get_transition_Energy(idx)
                        updated_mse = current_mse.clone()

                        if en_index == 4:
                            updated_mse[en_index] = transition_loss[i] * args.mse_weights
                        else:
                            updated_mse[en_index] = transition_loss[i]

                        mem.update_transition(idx, updated_mse)

                    if en_index == 0:
                        q_loss_tot = q_loss
                    else:
                        q_loss_tot += q_loss

                q_loss_tot = q_loss_tot / args.num_ensemble
                idx_tot = np.array(idx_tot).flatten()
                idx_tot = np.unique(idx_tot)

                # Update priorities of sampled transitions
                mem.update_priorities(idx_tot, q_loss_tot)

            if T % args.evaluation_interval == 0:
                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = ensemble_test(args, T, dqn_list, val_mem, metrics, results_dir,
                                                  num_ensemble=args.num_ensemble, scheduler=scheduler, action_p=action_p)  # Test
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))


                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].train()  # Set DQN (online network) back to training mode

                    # wandb.log({'eval/reward_mode': reward_mode_[T-1],
                    #            'eval/action_prob': action_probs_[T-1],
                    #            'eval/reward': reward,
                    #            'eval/Average_reward': avg_reward,
                    #            'eval/timestep': T,
                    #            'reliability/DQN': reliability[0],
                    #            'reliability/DDQN': reliability[1],
                    #            'reliability/Nosiy_DQN': reliability[2],
                    #            'reliability/Dueling_DQN': reliability[3],
                    #            'reliability/Distirbutional_DQN': reliability[4],
                    #            'Q-value/Q-value': avg_Q,
                    #            'Q-value/CE-loss': CE_loss,
                    #            'Q-value/batch-loss': batch_loss,
                    #            },step=T)

                # If memory path provided, save it
                if args.memory is not None:
                    save_memory(mem, args.memory, args.disable_bzip_memory)

            # Update target network
            if T % args.target_update == 0:
                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].update_target_net()

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].save(results_dir, 'checkpoint.pth')

        state = next_state

    env.close()