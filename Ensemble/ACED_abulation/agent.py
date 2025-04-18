# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F


from model import DQN, DDQN, NoisyDQN, DuelingDQN, DistributionalDQN, Rainbow

class Agent():
    def __init__(self, args, env, model, index=0): # shared memory
        self.action_space = env.action_space()
        self.index = index
        self.args = args
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.num_ensemble = args.num_ensemble
        self.loss_history = []

        #TODO: Q networks for each agents
        self.online_net = self.init_model(args, model, self.action_space).to(device=args.device)

        self.online_net.train()

        # self.target_net = DQN(args, self.action_space).to(device=args.device)

        #TODO: Q networks for each agents
        self.target_net = self.init_model(args, model, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)
        self.lr_schedular = CosineAnnealingWarmRestarts(self.optimiser, T_0=200000)

    def load_weights(self, block_id):
        model_path = 'ACED' + '/' + self.args.game
        model_path += '/Beta_' + str(self.args.beta_mean) + '_T_' + str(self.args.temperature)
        model_path += '_UCB_I_' + str(self.args.ucb_infer) + '_UCB_T_' + str(self.args.ucb_train) + '/'
        model_path += '/seed_' + str(self.args.seed) + '/'
        model_path += '/iteration_' + str(self.args.iteration) + '/'
        model_path += f'block_{block_id}_{self.index}th_model.pth'

        results_dir = os.path.join('./results', model_path)

        if os.path.isfile(results_dir):
            state_dict = torch.load(results_dir, map_location='cpu')
            if 'conv1.weight' in state_dict.keys():
                for old_key, new_key in (
                    ('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'),
                    ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'),
                    ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                    state_dict[new_key] = state_dict[old_key]
                    del state_dict[old_key]
            self.online_net.load_state_dict(state_dict)
            print(f"[Agent {self.index}] Loaded weights from block {block_id}")
        else:
            raise FileNotFoundError(f"Model not found: {results_dir}")

        #TODO: Modles for each agents
    def init_model(self, args, model, action_space):
        if model == 'DQN':
            return DQN(args, action_space)
        elif model == 'DDQN':
            return DDQN(args, action_space)
        elif model == 'NoisyDQN':
            return NoisyDQN(args, action_space)
        elif model == 'DuelingDQN':
            return DuelingDQN(args, action_space)
        elif model == 'DistributionalDQN':
            return DistributionalDQN(args, action_space)
        elif model == 'Rainbow':
            return Rainbow(args, action_space)
        else:
            raise ValueError("No known network")

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        if isinstance(self.online_net, (NoisyDQN, Rainbow)):
            self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            if isinstance(self.online_net, (DistributionalDQN, Rainbow)):
               return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
            else:
                return self.online_net(state.unsqueeze(0)).argmax(1).item()
    def act_v2(self, state):
        with torch.no_grad():
            if isinstance(self.online_net, (DistributionalDQN, Rainbow)):
                online_Q=(self.online_net(state.unsqueeze(0)) * self.support).sum(2)
                return online_Q, online_Q.argmax(1).item()
            else:
                online_Q = self.online_net(state.unsqueeze(0))
                return online_Q, online_Q.argmax(1).item()

    # Get Q-function
    def ensemble_q(self, state):
        with torch.no_grad():
            if isinstance(self.online_net, (DistributionalDQN, Rainbow)):
                return (self.online_net(state.unsqueeze(0)) * self.support).sum(2)
            else:
                return self.online_net(state.unsqueeze(0))

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)
    
    # Compute targe Q-value
    def get_target_q(self, next_states):
        with torch.no_grad():
            if isinstance(self.online_net, (DistributionalDQN, Rainbow)):
                pns = self.online_net(next_states)
                dns = self.support.expand_as(pns) * pns
                argmax_indices_ns = dns.sum(2).argmax(1)
                pns = self.target_net(next_states)
                pns_a = pns[range(self.batch_size), argmax_indices_ns]
                pns_a = pns_a * self.support.expand_as(pns_a)
                return pns_a.sum(1)
            if isinstance(self.online_net, (DDQN, DuelingDQN, NoisyDQN)):
                if isinstance(self.online_net, (NoisyDQN)):
                    self.target_net.reset_noise()
                next_actions = self.online_net(next_states).argmax(1)
                q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                return q_values
            else:
                q_values = self.target_net(next_states)
                return q_values.max(1).values

    def get_target_q_mse(self, next_state):
        if isinstance(self.online_net, (DistributionalDQN, Rainbow)):
            pns = self.online_net(next_state)
            dns = self.support.expand_as(pns) * pns
            argmax_index = dns.sum(2).argmax(1)
            pns = self.target_net(next_state)
            pns_a = pns.gather(1, argmax_index.unsqueeze(1).unsqueeze(2).expand(-1, 1, pns.size(2))).squeeze(1)
            q_values = torch.sum(self.support * pns_a)
            return q_values
        if isinstance(self.online_net, (DDQN, DuelingDQN, NoisyDQN)):
            next_action = self.online_net(next_state).argmax(1)
            q_values = self.target_net(next_state).gather(1, next_action.unsqueeze(-1)).squeeze(-1)
            return q_values
        else:
            q_values = self.target_net(next_state)
            return q_values.max(1).values


    
    # Compute Q-value
    def get_online_q(self, states):
        with torch.no_grad():
            if isinstance(self.online_net, (DistributionalDQN, Rainbow)):
                pns = self.online_net(states.unsqueeze(0))
                dns = self.support.expand_as(pns) * pns
                return dns.sum(2)
            else:
                q_values = self.online_net(states.unsqueeze(0))
                return q_values

    def diversity_learn(self, idxs, states, actions, returns, next_states, nonterminals, masks, weights, weight_Q=None):
        CE_loss = torch.tensor(0.0)

        if isinstance(self.online_net, (DistributionalDQN)):
            # Calculate current state probabilities (online network noise already sampled)
            ps = self.online_net(states)  # Probabilities p(s_t, ? ?online)
            ps_a_ = ps[range(self.batch_size), actions]  # p(s_t, a_t; ?online)

            q = torch.sum(self.support * ps_a_, dim=1)

            with torch.no_grad():
                pns_ = self.online_net(next_states)
                dns_ = self.support.expand_as(pns_) * pns_
                argmax_indices = dns_.sum(2).argmax(1)

                pns_ = self.target_net(next_states)
                pns_a_ = pns_[range(self.batch_size), argmax_indices]

                Q_target = torch.sum(self.support * pns_a_, dim=1)
                Q_t = returns + (nonterminals.squeeze() * (self.discount ** self.n) * Q_target)

            mse_loss = F.mse_loss(q, Q_t, reduction='none')
            batch_loss = (weights * masks * mse_loss).mean()
            transition_loss = (weights * masks * mse_loss)

            # Calculate current state probabilities (online network noise already sampled)
            log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
            log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

            with torch.no_grad():
                # Calculate nth next state probabilities
                pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
                dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
                argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
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
                (weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
                CE_loss = (weights * masks * loss).mean()
            else:
                (weight_Q * weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
                CE_loss = (weight_Q * weights * masks * loss).mean()
            self.optimiser.step()
            self.lr_schedular.step()


        elif isinstance(self.online_net, (DQN)):
            current_q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                # Max q value from target_net
                max_next_q_values = self.target_net(next_states).max(1)[0]
                # Calculate the target Q values : target_q = reward + (1 - done) * discount * max_next_q_values
                target_q_values = returns + (nonterminals.squeeze() * self.discount * max_next_q_values)

            loss = F.mse_loss(current_q_values, target_q_values, reduction='none')

            # Optimize the model
            self.online_net.zero_grad()
            if weight_Q is None:
                (weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
                batch_loss = (weights * masks * loss).mean()
                transition_loss = (weights * masks * loss)
            else:
                (weight_Q * weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
                batch_loss = (weight_Q * weights * masks * loss).mean()
                transition_loss = (weight_Q * weights * masks * loss)
            self.optimiser.step()
            self.lr_schedular.step()


        elif isinstance(self.online_net, (DDQN, DuelingDQN, NoisyDQN)):
            current_q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            if isinstance(self.online_net, (NoisyDQN)):
                self.target_net.reset_noise()  # Sample new target net noise
            with torch.no_grad():
                # Select actions with the highest Q-values for the next states using the online network (DDQN difference)
                next_actions = self.online_net(next_states).argmax(1)
                # Get the Q-values for those actions from the target network (DDQN difference)
                max_next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                # Calculate the target Q values
                target_q_values = returns + (nonterminals.squeeze() * self.discount * max_next_q_values)

            loss = F.mse_loss(current_q_values, target_q_values, reduction='none')

            # Optimize the model
            self.online_net.zero_grad()
            if weight_Q is None:
                (weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
                batch_loss = (weights * masks * loss).mean()
                transition_loss = (weights * masks * loss)
            else:
                (weight_Q * weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
                batch_loss = (weight_Q * weights * masks * loss).mean()
                transition_loss = (weight_Q * weights * masks * loss)
            self.optimiser.step()
            self.lr_schedular.step()

        return loss.detach().cpu().numpy(), batch_loss.detach().cpu().item(), CE_loss.detach().cpu().item(), transition_loss.detach().cpu()


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
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
        
    def ensemble_learn(self, idxs, states, actions, returns, next_states, nonterminals, weights, masks, weight_Q=None):
        # Calculate current state probabilities (online network noise already sampled)
        ps = self.online_net(states)  # Probabilities p(s_t, ? ?online)
        ps_a = ps[range(self.batch_size), actions]  # p(s_t, a_t; ?online)

        q = torch.sum(self.support * ps_a, dim=1)

        with torch.no_grad():
            pns = self.online_net(next_states)
            dns = self.support.expand_as(pns) * pns
            argmax_indices = dns.sum(2).argmax(1)

            self.target_net.reset_noise()
            pns = self.target_net(next_states)
            pns_a = pns[range(self.batch_size), argmax_indices]

            Q_target = torch.sum(self.support * pns_a, dim=1)
            Q_t = returns + (nonterminals.squeeze() * (self.discount ** self.n) * Q_target)

        m_loss = F.mse_loss(q, Q_t, reduction='none')
        if weight_Q is None:
            mse_loss = (weights * masks * m_loss)
        else:
            mse_loss = (weight_Q * weights * masks * m_loss)

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
            (weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
            batch_loss = (weights * masks * loss).mean()
        else:
            (weight_Q * weights * masks * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
            batch_loss = (weight_Q * weights * masks * loss).mean()
        self.optimiser.step()
        return loss.detach().cpu().numpy(), batch_loss.detach().cpu().item(), mse_loss.detach().cpu()


    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            if isinstance(self.online_net, (DistributionalDQN, Rainbow)):
               return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()
            else:
                return self.online_net(state.unsqueeze(0)).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()