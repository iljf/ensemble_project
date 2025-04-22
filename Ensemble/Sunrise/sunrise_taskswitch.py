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
from sunrise_memory import ReplayMemory
from test import ensemble_test, ensemble_test_taskswitch
from util_wrapper import *




def global_seed_initailizer(seed):
    # random seed for numpy
    np.random.seed(seed)
    # random seed for torch
    torch.manual_seed(seed)
    # random seed for torch.cuda
    torch.cuda.manual_seed(seed)


def predefined_scheduler_taskswitch(schedule_mode=1, action_prob_set = None, min_max_action_prob = [0.1, 0.9], max_length = 5000000, numstep_task = 500000, debug = False):

        # if env_name == 'breakout':
        #     reward_mode_info = {0: 'default', 1: 'reverse reward'}
        # elif env_name == 'freeway':
        #     reward_mode_info = {0: 'default', 1: 'negative reward for mistakes'}
        # elif env_name == 'space_invaders':
        #     reward_mode_info = {0: 'default', 1: 'kill gopher'}

        # if action_prob_set is None:
        #     action_prob_set = np.random.rand(4) * (min_max_action_prob[1] - min_max_action_prob[0]) + min_max_action_prob[0]
        # last iterations to be 0

        reward_mode_schedule = np.repeat(0, max_length)
        action_prob_seed_schedule = np.repeat(0, max_length)

        Envs= ['road_runner', 'freeway', 'space_invaders']

        num_env = max_length // numstep_task
        # randind for num_env

        # do not repeat the same env in a row
        randind = []
        for i in range(num_env):
            while True:
                idx = np.random.randint(0, len(Envs))
                if i == 0 or idx != randind[-1]:
                    randind.append(idx)
                    break

        Envs2out = []
        for i in range(num_env):
            # remeat 500000 times
            Envs2out.extend([Envs[randind[i]]]*numstep_task)

        return reward_mode_schedule, action_prob_seed_schedule, Envs2out


if __name__ == '__main__':


    # Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--id', type=str, default='sunrise_1e6', help='Experiment ID')
    parser.add_argument('--iteration', type=int, default=2, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=122, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument('--T-max', type=int, default=int(1.4e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='?', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='?', help='Prioritised experience replay exponent (originally denoted ?)')
    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='?', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='?', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(32000), metavar='?', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='?', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='?', help='Adam epsilon')
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
    parser.add_argument('--beta-mean', type=float, default=1.0, help='mean of bernoulli')
    parser.add_argument('--temperature', type=float, default=10, help='temperature for CF')
    parser.add_argument('--ucb-infer', type=float, default=1, help='coeff for UCB infer')
    parser.add_argument('--ucb-train', type=float, default=1, help='coeff for UCB train')
    parser.add_argument('--scheduler-mode', type=int, default=2, metavar='S', help='Scheduler seed/mode')
    parser.add_argument('--action-prob-max', type=float, default=0.5, help='max action probability')
    parser.add_argument('--action-prob-min', type=float, default=0.1, help='min action probability')
    # Setup
    args = parser.parse_args()

    # wandb intialize
    if args.id == 'sunrise_1e6':
        wandb.init(project="taskswitch",
                   name="taskswitch_SUNRISE_" + args.game + " " + "Seed" + str(args.seed) + "_B_" + str(args.beta_mean) + "_T_" + str(args.temperature) + "_UCB_I" + str(args.ucb_infer) + "_I_" + str(args.iteration),
                   config=args.__dict__
                   )

    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    # exp name
    exp_name = "taskswitch_" + args.id + '/' + args.game
    exp_name += '/Beta_' + str(args.beta_mean) + '_T_' + str(args.temperature)
    exp_name +='_UCB_I_' + str(args.ucb_infer) + '_UCB_T_' + str(args.ucb_train) + '/'
    exp_name += '/seed_' + str(args.seed) + '/'
    exp_name += str(args.iteration) + '/'

    results_dir = os.path.join('./results', exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(1, 10000))
    if torch.cuda.is_available() and not args.disable_cuda:
        args.device = torch.device('cuda:0')
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
    dqn_list = []
    for _ in range(args.num_ensemble):
        dqn = Agent(args, env)
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
    reward_mode_, action_probs_, Envs_ = predefined_scheduler_taskswitch(args.scheduler_mode, min_max_action_prob = [args.action_prob_min, args.action_prob_max], numstep_task=200000)
    # reward_mode_, action_probs_, info = predefined_scheduler(args.scheduler_mode, args.game, min_max_action_prob = [args.action_prob_min, args.action_prob_max], debug=True)

    # Construct validation memory
    val_mem = ReplayMemory(args, args.evaluation_size, args.beta_mean, args.num_ensemble)
    T, done = 0, True
    while T < args.evaluation_size:
        if done:
            state, done = env.reset(), False
        next_state, _, done = env.step(np.random.randint(0, action_space))
        val_mem.append(state, None, None, done)
        state = next_state
        T += 1




    if args.evaluate:
        for en_index in range(args.num_ensemble):
            dqn_list[en_index].eval()

        # KM: test code
        avg_reward, avg_Q, eval_index = ensemble_test_taskswitch(args, 0, dqn_list, val_mem, metrics, results_dir,
                                          num_ensemble=args.num_ensemble, evaluate=True)  # Test
        print('Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))
    else:
        # Training loop
        for en_index in range(args.num_ensemble):
            dqn_list[en_index].train()
        T, done = 0, True
        selected_en_index = np.random.randint(args.num_ensemble)

        while T < args.learn_start:
            if T % 10000 ==0:
                print(f"Running memory: {T}/{args.learn_start} for game: {args.game}")
            env.eps = action_probs_[T]
            env.env.reward_mode = reward_mode_[T]
            action_p = env.eps
            scheduler = env.env.reward_mode

            if done:
                state, done = env.reset(), False
                selected_en_index = np.random.randint(args.num_ensemble)

            if T % args.replay_frequency == 0:
                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].reset_noise()  # Draw a new set of noisy weights

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
            else:
                action = dqn_list[selected_en_index].act(state)  # Choose an action greedily (with noisy weights)
            next_state, reward, done = env.step(action)  # Step
            # scheduler.update(T)
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            mem.append(state, action, reward, done)  # Append transition to memory
            state = next_state
            T += 1

        log('Memory:' + ' T = ' + str(T) + ' / ' + str(args.learn_start))

        # Set reward mode, action prob according to the schedule
        for T in trange(1, args.T_max + 1):
            game_bf = args.game
            args.game = Envs_[T-1]
            game_af = args.game
            if game_bf != game_af:
                env = Env(args)
                env = Rewardvalue(env)
                env = Action_random(env, eps=0.1)
                env.train()
                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].action_space = env.action_space()
            env.eps = action_probs_[T-1]
            env.env.reward_mode = reward_mode_[T-1]
            action_p = env.eps
            scheduler = env.env.reward_mode

            if done:
                state, done = env.reset(), False
                selected_en_index = np.random.randint(args.num_ensemble)

            if T % args.replay_frequency == 0:
                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].reset_noise()  # Draw a new set of noisy weights

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
            else:
                action = dqn_list[selected_en_index].act(state)  # Choose an action greedily (with noisy weights)
            # clipping action to env.action_space()
            action = max(min(action, env.action_space() - 1), 0)
            next_state, reward, done = env.step(action)  # Step
            # scheduler.update(T)
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            mem.append(state, action, reward, done)  # Append transition to memory

            # Train and test
            # if T >= args.learn_start:
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight ? to 1
            if T % args.replay_frequency == 0:
                total_q_loss = 0

                # Sample transitions
                idxs, states, actions, returns, next_states, nonterminals, weights, masks = mem.sample(args.batch_size)
                q_loss_tot = 0

                weight_Q = None
                # Corrective feedback
                if args.temperature > 0:
                    mean_Q, var_Q = None, None
                    L_target_Q = []
                    for en_index in range(args.num_ensemble):
                        target_Q = dqn_list[en_index].get_target_q(next_states)
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

                    # std_Q max
                    std_Q_max = max(std_Q)
                    # std_Q min
                    std_Q_min = min(std_Q)
                    # std_Q mean
                    std_Q_mean = sum(std_Q) / len(std_Q)

                    weight_Q = torch.sigmoid(-std_Q*args.temperature) + 0.5


                for en_index in range(args.num_ensemble):
                    # Train with n-step distributional double-Q learning
                    q_loss, batch_loss = dqn_list[en_index].ensemble_learn(idxs, states, actions, returns,
                                                               next_states, nonterminals, weights,
                                                               masks[:, en_index], weight_Q)
                    if en_index == 0:
                        q_loss_tot = q_loss
                    else:
                        q_loss_tot += q_loss
                q_loss_tot = q_loss_tot / args.num_ensemble

                # Update priorities of sampled transitions
                mem.update_priorities(idxs, q_loss_tot)

            if T % args.evaluation_interval == 0:
                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].eval()  # Set DQN (online network) to evaluation mode
                avg_reward, avg_Q = ensemble_test_taskswitch(args, T, dqn_list, val_mem, metrics, results_dir, num_ensemble=args.num_ensemble, scheduler=scheduler, action_p=action_p)  # Test
                log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q))


                for en_index in range(args.num_ensemble):
                    dqn_list[en_index].train()  # Set DQN (online network) back to training mode

                    wandb.log({'eval/reward_mode': reward_mode_[T-1],
                               'eval/action_prob': action_probs_[T-1],
                               'eval/reward': reward,
                               'eval/Average_reward': avg_reward,
                               'eval/timestep': T,
                               'Q-value/Q-value': avg_Q,
                               'Q-value/batch-loss': batch_loss,
                               'Q-value/batch-std-Q-mean': std_Q_mean,
                               'Q-value/batch-std-Q-min': std_Q_min,
                               'Q-value/batch-std-Q-max': std_Q_max,
                               },step=T)

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