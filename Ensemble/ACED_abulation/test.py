# -*- coding: utf-8 -*-
from __future__ import division
import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import numpy as np
import wandb

from util_wrapper import Rewardvalue, Action_random
from env import Env
import torch.nn.functional as F
from memory import ReplayMemory


# Test DQN
def test(args, T, dqn, val_mem, metrics, results_dir, evaluate=False):
    env = Env(args)
    env.eval()
    metrics['steps'].append(T)
    T_rewards, T_Qs = [], []

    # Test performance over several episodes
    done = True
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
            dqn.save(results_dir)

        # Append to results and save metrics
        metrics['rewards'].append(T_rewards)
        metrics['Qs'].append(T_Qs)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Plot
        _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)
        _plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)

    # Return average reward and Q-value
    return avg_reward, avg_Q

def ensemble_test(args, T, dqn, val_mem, metrics, results_dir, num_ensemble, evaluate=False, scheduler=None, action_p=None, memory=None):
    env = Env(args)
    env = Rewardvalue(env)
    env = Action_random(env, eps=0.1)
    env.eval()
    metrics['steps'].append(T)

    env.eps = env.eps
    env.env.reward_mode = env.env.reward_mode

    T_rewards, T_Qs = [], []
    T_mse = []
    ep_reliability = []
    # Test performance over several episodes
    done = True

    if args.evaluate:
        n_episodes = 50
        T_episodes = 30
        reliability = torch.ones(args.num_ensemble, device=args.device) / args.num_ensemble

        for episode_total in range(T_episodes):
            print(f"Total episode {episode_total + 1}/{T_episodes}")
            reward_mode = scheduler
            action_probs = action_p
            env.env.reward_mode = reward_mode
            env.eps = action_probs
            mse_log = []
            ep_rewards = []

            for ep in range(n_episodes):
                print(f"Running episode {ep + 1}/{n_episodes} for game: {args.game}")
                while True:
                    if done:
                        state, reward_sum, done = env.reset(), 0, False
                    # average agent selection
                    if args.permutation == 1:
                        Q_list = []
                        reliability = np.random.uniform(0.2,0.2, size=num_ensemble)
                        for en_index in range(num_ensemble):
                            online_Q = dqn[en_index].ensemble_q(state)
                            Q_list.append(online_Q)
                        Q_tot = sum(Q_list[i] * reliability[i] for i in range(len(Q_list)))
                        action = Q_tot.argmax(1).item()

                        state, reward, done = env.step(action)
                        reward_sum += reward
                        # if done or T >= max_steps:
                        if done:
                            T_rewards.append(reward_sum)
                            break
                    # random agent selection
                    if args.permutation == 2:
                        random_agent = np.random.randint(0, num_ensemble)
                        q_values = dqn[random_agent].ensemble_q(state)
                        action = q_values.argmax(1).item()

                        state, reward, done = env.step(action)
                        reward_sum += reward
                        if done:
                            T_rewards.append(reward_sum)
                            break
                    # mse_log
                    if args.permutation == 3:
                        Q_list = []
                        q_tot = 0
                        for en_index in range(num_ensemble):
                            online_Q = dqn[en_index].ensemble_q(state)
                            Q_list.append(online_Q)
                            if en_index == 0:
                                q_tot = online_Q
                            else:
                                q_tot += online_Q
                        action = q_tot.argmax(1).item()
                        next_state, reward, done = env.step(action)
                        if memory is not None:
                            nonterminal = 1.0 if not done else 0.0
                            mse_list = []
                            for en_index in range(num_ensemble):
                                online_Q = Q_list[en_index][0, action]
                                target_Q = dqn[en_index].get_target_q_mse(state)
                                target_Q = reward + (nonterminal * args.discount * target_Q)
                                target_Q = torch.flatten(target_Q)
                                online_Q = torch.flatten(online_Q)
                                mse_loss = F.mse_loss(online_Q, target_Q)
                                mse_list.append(mse_loss)
                            mse_list[-1] = args.mse_weights * mse_list[-1]  # dist-DQN
                            mse_tensor = torch.stack(mse_list)
                            # memory.append(state, action, reward, done, mse_tensor)
                            if reward == 0:
                                mse_log.append(mse_tensor.cpu().numpy().tolist())

                        state = next_state
                        reward_sum += reward
                        if args.render:
                            env.render()
                        # if done or T >= max_steps:
                        if done:
                            T_rewards.append(reward_sum)
                            break
                    # UCB
                    if args.permutation == 4:
                        mean_Q, var_Q = None, None
                        L_target_Q = []
                        for en_index in range(args.num_ensemble):
                            target_Q = dqn[en_index].ensemble_q(state)
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

                        state, reward, done = env.step(action)
                        reward_sum += reward
                        if done:
                            T_rewards.append(reward_sum)
                            break
                    # ACED-DQN
                    else:
                        Q_list = []
                        for en_index in range(num_ensemble):
                            q_values = dqn[en_index].get_online_q(state)
                            Q_list.append(q_values)

                        Q_tot = sum(Q_list[i] * reliability[i] for i in range(len(Q_list)))
                        action = Q_tot.argmax().item()
                        next_state, reward, done = env.step(action)

                        nonterminal = 1.0 if not done else 0.0
                        mse_list = []
                        for en_index in range(num_ensemble):
                            online_Q = Q_list[en_index][0, action]
                            target_Q = dqn[en_index].get_target_q_mse(next_state)
                            target_Q = reward + nonterminal * args.discount * target_Q

                            mse_loss = F.mse_loss(online_Q.flatten(), target_Q.flatten())
                            mse_list.append(mse_loss)
                        mse_list[-1] = args.mse_weights * mse_list[-1]
                        mse_tensor = torch.stack(mse_list)

                        # reliability update
                        softmax_reliability = F.softmax(-mse_tensor / args.mse_temperature, dim=0)
                        momentum_reliability = (1 - args.gamma) * reliability + args.gamma * softmax_reliability
                        momentum_reliability = torch.clamp(momentum_reliability, min=0.2, max=0.5)
                        reliability = momentum_reliability / momentum_reliability.sum()

                        state = next_state
                        reward_sum += reward
                        # step = args.block_id * n_episodes + ep + 1

                        if done:
                            ep_rewards.append(reward_sum)
                            # wandb.log({
                            #     'eval/reward': ep_rewards[-1],
                            #     'eval/block': args.block_id,
                            #     'reliability/DQN': reliability[0].item(),
                            #     'reliability/DDQN': reliability[1].item(),
                            #     'reliability/NoisyDQN': reliability[2].item(),
                            #     'reliability/DuelingDQN': reliability[3].item(),
                            #     'reliability/DistributionalDQN': reliability[4].item(),
                            #     'mse/DQN': mse_tensor[0].item(),
                            #     'mse/DDQN': mse_tensor[1].item(),
                            #     'mse/NoisyDQN': mse_tensor[2].item(),
                            #     'mse/DuelingDQN': mse_tensor[3].item(),
                            #     'mse/DistributionalDQN': mse_tensor[4].item(),
                            # }, step=step)
                            break

            env.close()
            ep_reliability.append(reliability.cpu().numpy().tolist())
            T_mse.append(mse_tensor.cpu().numpy().tolist())
            T_rewards.append(ep_rewards[-1])

            for state in val_mem:  # Iterate over valid states
                for en_index in range(num_ensemble):
                    T_Qs.append(dqn[en_index].evaluate_q(state))

        if args.permutation == 3:
            avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
            return avg_reward, avg_Q, T_rewards, mse_log
        else:
            avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
            return avg_reward, avg_Q, T_rewards, ep_reliability, T_mse
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
                q_tot = 0
                for en_index in range(num_ensemble):
                    if en_index == 0:
                        q_tot = dqn[en_index].ensemble_q(state)
                    else:
                        q_tot += dqn[en_index].ensemble_q(state)
                action = q_tot.argmax(1).item()

                state, reward, done = env.step(action)  # Step
                reward_sum += reward
                if args.render:
                    env.render()
                # if done or T >= max_steps:
                if done:
                    T_rewards.append(reward_sum)
                    break
        env.close()

        # Test Q-values over validation memory
        for state in val_mem:  # Iterate over valid states
            for en_index in range(num_ensemble):
                T_Qs.append(dqn[en_index].evaluate_q(state))

        avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
        if not evaluate:
            # Save model parameters if improved
            if avg_reward > metrics['best_avg_reward']:
                metrics['best_avg_reward'] = avg_reward
                for en_index in range(num_ensemble):
                    dqn[en_index].save(results_dir, name='%dth_model.pth'%(en_index))

            # Append to results and save metrics
            metrics['rewards'].append(T_rewards)
            metrics['Qs'].append(T_Qs)

            torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))
            # Plot
            _plot_line(metrics['steps'], metrics['rewards'], 'Reward', path=results_dir)
            _plot_line(metrics['steps'], metrics['Qs'], 'Q', path=results_dir)

            if T % 200000 == 0:
                block_id = (T // 200000)-1
                for en_index in range(num_ensemble):
                    dqn[en_index].save(results_dir, name='block_%d_%dth_model.pth' % (block_id, en_index))
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