# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from model import DQN, DQNV, DDQN, NoisyDQN, DuelingDQN, DistributionalDQN

class Agent():
    def __init__(self, args, env, model):
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.model = args.model_name

        # self.online_net = DQN(args, self.action_space).to(device=args.device)

        #TODO: Q networks for each agents
        self.online_net = self.init_model(args, model, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                        del state_dict[old_key]  # Delete old keys for strict load_state_dict
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        # self.target_net = DQN(args, self.action_space).to(device=args.device)

        #TODO: Q networks for each agents
        self.target_net = self.init_model(args, model, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

        #TODO: Modles for each agents
    def init_model(self, args, model, action_space):
        if model == 'DQNV':
            return DQNV(args, action_space)
        elif model == 'DDQN':
            return DDQN(args, action_space)
        elif model == 'NoisyDQN':
            return NoisyDQN(args, action_space)
        elif model == 'DuelingDQN':
            return DuelingDQN(args, action_space)
        elif model == 'DistributionalDQN':
            return DistributionalDQN(args, action_space)
        else:
            raise ValueError("wtf")
    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        if self.model == 'NoisyDQN':
            self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            if self.model == 'DistributionalDQN':
                return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
            else:
                return self.online_net(state.unsqueeze(0)).argmax(1).item()
    # Get Q-function
    def ensemble_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2)
        
    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)
    
    # Compute targe Q-value
    def get_target_q(self, next_states):
        with torch.no_grad():
            if self.model == 'DistributionalDQN':
                pns = self.online_net(next_states)
                dns = self.support.expand_as(pns) * pns
                argmax_indices_ns = dns.sum(2).argmax(1)
                pns = self.target_net(next_states)
                pns_a = pns[range(self.batch_size), argmax_indices_ns]
                pns_a = pns_a * self.support.expand_as(pns_a)
                return pns_a.sum(1)
            else:
                q_values = self.target_net(next_states)
                return q_values.argmax(1)

    # Compute Q-value
    def get_online_q(self, states):
        with torch.no_grad():
            if self.model == 'DistributionalDQN':
                pns = self.online_net(states.unsqueeze(0))
                dns = self.support.expand_as(pns) * pns
                return dns.sum(2)
            else:
                q_values = self.online_net(states.unsqueeze(0))
                return q_values

    def DQNV_learn(self, mem):
        # Sample transitions
        idxs, states, actions, rewards, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Get current Q estimates from the online network
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        nonterminals = nonterminals.squeeze(-1)

        with torch.no_grad():
            # DQN에서는 target_net에서 최대 Q 값을 바로 가져옴
            max_next_q_values = self.target_net(next_states).max(1)[0]

            # Calculate the target Q values
            target_q_values = rewards + (nonterminals * self.discount * max_next_q_values)

        # Calculate the loss as the mean squared error between current and target Q values
        loss = [F.mse_loss(current_q_values[i], target_q_values[i]) for i in range(len(current_q_values))]
        loss = torch.stack(loss)

        # Optimize the model
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        # Convert the loss to an array and replicate it across the batch
        # loss_array = np.full(len(idxs), loss.detach().cpu().item())

        # Optionally, update priorities in the memory (if using prioritized experience replay)
        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def DDQN_learn(self, mem):
        # Sample transitions
        idxs, states, actions, rewards, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Get current Q estimates from the online network
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Select actions with the highest Q-values for the next states using the online network (DDQN difference)
            next_actions = self.online_net(next_states).argmax(1)
            # Get the Q-values for those actions from the target network (DDQN difference)
            max_next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
            # Calculate the target Q values
            target_q_values = rewards + (nonterminals.squeeze() * self.discount * max_next_q_values)

        td_error = target_q_values - current_q_values
        loss = torch.abs(td_error)

        # Optimize the model
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        # Optionally, update priorities in the memory (if using prioritized experience replay)
        mem.update_priorities(idxs, loss.detach().cpu().numpy())


    def learn_by_name(self, mem, model_name):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        if model_name == "DistributionalDQN":
            # Calculate current state probabilities (online network noise already sampled)
            log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
            log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

            with torch.no_grad():
                # Calculate nth next state probabilities
                pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
                dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
                argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
                # self.target_net.reset_noise()  # Sample new target net noise
                pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
                pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

                # Compute Tz (Bellman operator T applied to z)
                Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
                Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
                # Compute L2 projection of Tz onto fixed support z
                b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
                l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
                # Fix disappearing probability mass when l = b = u (b is int)
                l[(u > 0) * (l == u)] -= 1
                u[(l < (self.atoms - 1)) * (l == u)] += 1

                # Distribute probability of Tz
                m = states.new_zeros(self.batch_size, self.atoms)
                offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
                m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
                m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

            loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
            self.online_net.zero_grad()
            (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
            batch_loss = (weights * loss).mean()
            self.optimiser.step()

            mem.update_priorities(idxs, loss.detach().cpu().numpy())

        elif model_name == "DQNV":
            # Get current Q estimates from the online network
            current_q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                # Max q value from target_net
                max_next_q_values = self.target_net(next_states).max(1)[0]
                # Calculate the target Q values : target_q = reward + (1 - done) * discount * max_next_q_values
                target_q_values = returns + (nonterminals.squeeze() * self.discount * max_next_q_values)

            # Prioirity selection 때 np.power 로 절대값으로 변환하기에 mse loss 사용
            loss = [F.mse_loss(current_q_values[i], target_q_values[i]) for i in range(len(current_q_values))]
            loss = torch.stack(loss)
            # td_error = target_q_values - current_q_values  # L1 loss
            # loss = torch.abs(td_error)  # torch.abs computes the absolute value of each element.

            # Optimize the model
            self.online_net.zero_grad()
            (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
            batch_loss = (weights * loss).mean()
            self.optimiser.step()

            mem.update_priorities(idxs, loss.detach().cpu().numpy())

        elif model_name in ('DDQN', 'DuelingDQN', 'NoisyDQN'):
            # Get current Q estimates from the online network
            current_q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                # Select actions with the highest Q-values for the next states using the online network (DDQN difference)
                next_actions = self.online_net(next_states).argmax(1)
                # Get the Q-values for those actions from the target network (DDQN difference)
                max_next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                # Calculate the target Q values
                target_q_values = returns + (nonterminals.squeeze() * self.discount * max_next_q_values)

            # Prioirity selection 때 np.power 로 절대값으로 변환하기에 mse loss 사용
            loss = [F.mse_loss(current_q_values[i], target_q_values[i]) for i in range(len(current_q_values))]
            loss = torch.stack(loss)
            # td_error = target_q_values - current_q_values
            # loss = torch.abs(td_error)

            # Optimize the model
            self.online_net.zero_grad()
            (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
            batch_loss = (weights * loss).mean()
            self.optimiser.step()

            mem.update_priorities(idxs, loss.detach().cpu().numpy())

        return batch_loss.detach().cpu().item()

    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)


        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            # self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
        
    def ensemble_learn(self, idxs, states, actions, returns, next_states, nonterminals, weights, masks, weight_Q=None):
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

        self.online_net.zero_grad()
        if weight_Q is None:
            (weights* masks * loss).mean().backward() # Backpropagate importance-weighted minibatch loss
            batch_loss = (weights * masks * loss).mean()
        else:
            (weight_Q * weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
            batch_loss = (weight_Q * weights * masks * loss).mean()
        self.optimiser.step()
        return loss.detach().cpu().numpy(), batch_loss.detach().cpu().item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            if self.model == 'DistributionalDQN':
               return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()
            else:
                return self.online_net(state.unsqueeze(0)).argmax(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()