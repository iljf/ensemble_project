# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import os
import pickle
import wandb

import atari_py
import numpy as np
import torch
from tqdm import trange

from agent import Agent
from env import Env
from memory import ReplayMemory
from test import test, ensemble_test
from util_wrapper import *




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
        # if action_prob_set is None:
        #     action_prob_set = np.random.rand(4) * (min_max_action_prob[1] - min_max_action_prob[0]) + min_max_action_prob[0]
        # last iterations to be 0
        if action_prob_set is None:
            action_prob_set = np.random.rand(3) * (min_max_action_prob[1] - min_max_action_prob[0]) + min_max_action_prob[0]

        else:
            if len(action_prob_set) != 4:
                raise ValueError('action_prob_set should be of length 4')


        """
        reward mode sampling 할때 마지막 400k 에서 500k를 0으로 세팅할때
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
            # last 100k to be 0
            action_prob_seed = np.append(action_prob_seed, 0)
            # repeat each of them 100k times

            action_prob_seed_schedule = np.repeat(action_prob_seed, 200000)

            # TODO check the code;
            # repeat by 100k times till 400k
            # action_mode_seed = (action_prob_seed * num_repeats)
            # action_prob_seed_schedule = np.repeat(action_mode_seed, 100000)

        else: # if schedule_mode is 1,3,5,7 then continuous
            action_prob_seed_schedule = np.random.rand(1000000)/5

        if debug:
            rand_cond_seed = [ [j for _ in range((5-1)//len(reward_mode_info.keys()))] for j in range(len(reward_mode_info.keys()))]
            rand_cond_seed = np.array(rand_cond_seed).flatten()

            # # random shuffle of the predefined reward modes
            np.random.shuffle(rand_cond_seed)
            rand_cond_seed = np.append(1, rand_cond_seed)

            # repeat each of them 100k times
            reward_mode_schedule = np.repeat(rand_cond_seed, 200000)


        return reward_mode_schedule, action_prob_seed_schedule, reward_mode_info


if __name__ == '__main__':


    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--id', type=str, default='block_rainbow_ca', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=122, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    # parser.add_argument('--model_name', type=str, default='DistributionalDQN', help='Models of Q networks')
    parser.add_argument('--model_name', type=str, default='NoisyDQN', help='Models of Q networks = [DQNV, DDQN, NoisyDQN, DuelingDQN, DistributionalDQN]')
    parser.add_argument('--game', type=str, default='frostbite', choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument('--T-max', type=int, default=int(10e5), metavar='STEPS', help='Number of training steps (4x number of frames)')
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
    parser.add_argument('--target-update', type=int, default=int(32000), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
    parser.add_argument('--learn-start', type=int, default=int(80000), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=1000, metavar='STEPS', help='Number of training steps between evaluations')
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
    parser.add_argument('--temperature', type=float, default=10, help='temperature for CF')
    parser.add_argument('--ucb-infer', type=float, default=1, help='coeff for UCB infer')
    parser.add_argument('--ucb-train', type=float, default=1, help='coeff for UCB train')
    parser.add_argument('--scheduler-mode', type=int, default=2, metavar='S', help='Scheduler seed/mode')
    parser.add_argument('--action-prob-max', type=float, default=0.5, help='max action probability')
    parser.add_argument('--action-prob-min', type=float, default=0.1, help='min action probability')
    parser.add_argument('--block-id', type=int, default=0, help='testing schedule block')

    # Setup
    args = parser.parse_args()

    if args.id == 'Diverse_rainbow':
        wandb.init(project="diverse_rainbow",
                   name=args.model_name + " " + args.game + " " + "Seed" + str(args.seed),
                   config=args.__dict__
                   )
    elif args.id == 'block_rainbow_ca':
        wandb.init(project="blt",
               name=args.model_name + "_r_ " + args.game + "_b_" + str(args.block_id) + "_Seed" + str(args.seed),
               config=args.__dict__
               )

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    # exp name
    exp_name = args.id + '_' + str(args.num_ensemble) + '/' + args.game

    results_dir = os.path.join('./results', exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

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
    model = args.model_name
    dqn = Agent(args, env, model)


    # If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
    if args.model is not None and not args.evaluate:
        if not args.memory:
            raise ValueError('Cannot resume training without memory save path. Aborting...')
        elif not os.path.exists(args.memory):
            raise ValueError('Could not find memory file at {path}. Aborting...'.format(path=args.memory))
        mem = load_memory(args.memory, args.disable_bzip_memory)
    else:
        mem = ReplayMemory(args, args.memory_capacity)

    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


    # scheduler
    global_seed_initailizer(args.seed)
    reward_mode_, action_probs_, info = predefined_scheduler(args.scheduler_mode, args.game, min_max_action_prob = [args.action_prob_min, args.action_prob_max])
    block_id = args.block_id # 0 = 0~100k, 1 = 100k~200k, 2 = 200k~300k
    reward_mode_, action_probs_ = reward_mode_[(block_id)*int(20e4):(block_id+1)*int(20e4)], action_probs_[(block_id)*int(20e4):(block_id+1)*int(20e4)]
    # reward_mode_, action_probs_, info = predefined_scheduler(args.scheduler_mode, args.game, min_max_action_prob = [args.action_prob_min, args.action_prob_max], debug=True)

    # Construct validation memory
    val_mem = ReplayMemory(args, args.evaluation_size)
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            state, done = env.reset(), False
        next_state, _, done = env.step(np.random.randint(0, action_space))
        val_mem.append(state, None, None, done)
        state = next_state
        T += 1




    if args.evaluate:
        dqn.eval()
        avg_reward, avg_Q = test(args, 0, dqn, val_mem, metrics, results_dir, evaluate=True)  # Test
        print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
    else:
        # Training loop
        dqn.train()
        T, done = 0, True

        while T < args.learn_start:
            env.eps = action_probs_[T]
            env.env.reward_mode = reward_mode_[T]
            action_p = env.eps
            scheduler = env.env.reward_mode

            if done:
                state, done = env.reset(), False

            if T % args.replay_frequency == 0 and args.model_name == 'NoisyDQN':
                dqn.reset_noise()  # Draw a new set of noisy weights
            action = dqn.act(state)  # Choose an action greedily (with noisy weights)
            next_state, reward, done = env.step(action)  # Step

            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards

            mem.append(state, action, reward, done)  # Append transition to memory
            state = next_state
            T += 1
        log('Memory:' + ' T = ' + str(T) + ' / ' + str(args.learn_start))
        # Set reward mode, action prob according to the schedule
        for T in trange(1, args.T_max + 1):
            env.eps = action_probs_[T-1]
            env.env.reward_mode = reward_mode_[T-1]
            action_p = env.eps
            scheduler = env.env.reward_mode

            if done:
                state, done = env.reset(), False

            if T % args.replay_frequency == 0 and args.model_name == 'NoisyDQN':
                dqn.reset_noise()  # Draw a new set of noisy weights

            action = dqn.act(state)  # Choose an action greedily (with noisy weights)
            next_state, reward, done = env.step(action)  # Step
            # scheduler.update(T)
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            mem.append(state, action, reward, done)  # Append transition to memory

            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1
            # if T % args.replay_frequency == 0 and args.model_name == 'DQNV':
            #     dqn.DQNV_learn(mem)  # Train DQN
            # elif T % args.replay_frequency == 0 and args.model_name == 'DDQN':
            #     dqn.DDQN_learn(mem)  # Train DDQN
            # elif T % args.replay_frequency == 0 and args.model_name == 'DuelingDQN':
            #     dqn.DDQN_learn(mem)
            # elif T % args.replay_frequency == 0 and args.model_name == 'NoisyDQN':
            #     dqn.DDQN_learn(mem)
            # elif T % args.replay_frequency == 0 and args.model_name == 'DistributionalDQN':
            if T % args.replay_frequency == 0:
                batch_loss = dqn.learn_by_name(mem, args.model_name)




            if T % args.evaluation_interval == 0:
                dqn.eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = test(args, T, dqn, val_mem, metrics, results_dir, scheduler=scheduler, action_p=action_p)  # Test
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
                dqn.train()

                wandb.log({'eval/reward_mode': reward_mode_[T-1],
                            'eval/action_prob': action_probs_[T-1],
                            'eval/reward': reward,
                            'eval/Average_reward': avg_reward,
                            'eval/timestep': T,
                            'Q-value/Q-value': avg_Q,
                            'Q-value/batch-loss': batch_loss
                             },step=T)

                # If memory path provided, save it
                if args.memory is not None:
                    save_memory(mem, args.memory, args.disable_bzip_memory)

            # Update target network
            if T % args.target_update == 0:
                dqn.update_target_net()

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                dqn.save(results_dir, 'checkpoint.pth')

        state = next_state

    env.close()