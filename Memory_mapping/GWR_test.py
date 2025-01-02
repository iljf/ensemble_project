# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import pandas as pd
from GWR_agent import Agent
import numpy as np

from util_wrapper import Rewardvalue, Action_random
from env import Env

# Test DQN
def test(args, T, dqn, val_mem, metrics, results_dir, evaluate=False, scheduler=None, action_p=None):
    env = Env(args)
    env = Rewardvalue(env)
    env = Action_random(env, eps=0.1)
    env.eval()
    metrics['steps'].append(T)
    T_rewards, T_Qs = [], []

    env.eps = env.eps
    env.env.reward_mode = env.env.reward_mode

    # Test performance over several episodes
    done = True

    if args.evaluate:
        n_episodes = 100
        for episode_num in range(n_episodes):
            reward_mode = scheduler
            action_probs = action_p
            env.env.reward_mode = reward_mode
            env.eps = action_probs

        for _ in range(n_episodes):
            print(f"Running episode {_ + 1}/{n_episodes} for model: {args.model_name}, block_id: {args.block_id}")
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
                state, reward, done = env.step(action)  # Step
                reward_sum += reward
                if args.render:
                    env.render()
                if done:
                    T_rewards.append(reward_sum)
                    break
        env.close()

        # Test Q-values over validation memory
        for state in val_mem:  # Iterate over valid states
            T_Qs.append(dqn.evaluate_q(state))

        avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
        print(f"model: {args.model_name}, block_id: {args.block_id}, avg_r: {avg_reward}, avg_q: {avg_Q}")

            # Return average reward and Q-value
        return avg_reward, avg_Q

    else:
        for episode_num in range(args.evaluation_episodes):
            reward_mode = scheduler
            action_probs = action_p
            env.env.reward_mode = reward_mode
            env.eps = action_probs

        for _ in range(args.evaluation_episodes):
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
                state, reward, done = env.step(action)  # Step
                reward_sum += reward
                if args.render:
                    env.render()
                if done:
                    T_rewards.append(reward_sum)
                    break
        env.close()

        # Test Q-values over validation memory
        for state in val_mem:  # Iterate over valid states
            T_Qs.append(dqn.evaluate_q(state))

        avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)

        if not evaluate:
            # Save model parameters if improved
            if avg_reward > metrics['best_avg_reward']:
                metrics['best_avg_reward'] = avg_reward
                dqn.save(results_dir, name=f'{args.block_id}_model.pth')

            # Append to results and save metrics
            metrics['rewards'].append(T_rewards)
            metrics['Qs'].append(T_Qs)
            torch.save(metrics, os.path.join(results_dir, f'{args.block_id}_metrics.pth'))

            # Plot
            _plot_line(metrics['steps'], metrics['rewards'], f'{args.block_id}_Reward', path=results_dir)
            _plot_line(metrics['steps'], metrics['Qs'], f'{args.block_id}_Q', path=results_dir)

            # Return average reward and Q-value
        return avg_reward, avg_Q

# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    ys = torch.tensor(ys_population, dtype=torch.float32)
    ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
    ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

    trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
    trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
    trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
    trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
    trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

    plotly.offline.plot({
        'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
        'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)