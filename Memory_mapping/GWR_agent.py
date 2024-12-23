from __future__ import division
import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from model import DQN, DDQN, NoisyDQN, DuelingDQN

class Agent:
    def __init__(self, args, env, model):
        self.action_space = env.action_space()
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.model = args.model_name

        self.online_net = self.init_model(args, model, self.action_space).to(device=args.device)

        if args.evaluate:
            model_pth = args.id + '/' + args.game + '/' + args.model_name + '/'
            dir = os.path.join('./results', model_pth)
            file = f'{args.block_id}_model.pth'
            path = os.path.join(dir, file)

            if os.path.isfile(path):
                state_dict = torch.load(path, map_location='cpu')
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (
                        ('conv1.weight', 'convs.0.weight'),
                        ('conv1.bias', 'convs.0.bias'),
                        ('conv2.weight', 'convs.2.weight'),
                        ('conv2.bias', 'convs.2.bias'),
                        ('conv3.weight', 'convs.4.weight'),
                        ('conv3.bias', 'convs.4.bias')
                    ):
                        state_dict[new_key] = state_dict[old_key]
                        del state_dict[old_key]
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + path)
            else:
                raise FileNotFoundError(path)

        self.online_net.train()

        self.target_net = self.init_model(args, model, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = torch.optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    def init_model(self, args, model, action_space):
        if model == 'DQN':
            return DQN(args, action_space)
        elif model == 'DDQN':
            return DDQN(args, action_space)
        elif model == 'NoisyDQN':
            return NoisyDQN(args, action_space)
        elif model == 'DuelingDQN':
            return DuelingDQN(args, action_space)

    def GWR_learn(self, mem, model_name):
        states, actions, rewards, next_states, nonterminals = mem.sample(self.batch_size)

        states = states.float() / 255.0
        next_states = next_states.float() / 255.0

        if model_name == 'DQN':
            current_q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_actions = self.online_net(next_states).argmax(1)
                max_next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                target_q_values = rewards + (nonterminals.squeeze() * self.discount * max_next_q_values)
            loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
            loss = loss.mean()

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

        elif model_name in ('DDQN', 'NoisyDQN', 'DuelingDQN'):
            current_q_values = self.online_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_actions = self.online_net(next_states).argmax(1)
                max_next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                target_q_values = rewards + (nonterminals.squeeze() * self.discount * max_next_q_values)

            loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
            loss = loss.mean()

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()


        return loss.detach().cpu().item()

    def compute_loss(self, states, returns, actions):
        q_values = self.online_net(states)
        td_error = returns - q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = (td_error ** 2).mean()
        return loss

    def reset_noise(self):
        if self.model in ('NoisyDQN'):
            self.online_net.reset_noise()

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def act(self, state):
        with torch.no_grad():
            if self.model in ('DistributionalDQN', 'Rainbow'):
                return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
            else:
                return self.online_net(state.unsqueeze(0)).argmax(1).item()

    def evaluate_q(self, state):
        state = state.to(next(self.online_net.parameters()).device)
        return self.online_net(state.unsqueeze(0)).max(1)[0].item()

    def act_e_greedy_lr(self, state, epsilon=1.0):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    def act_e_greedy(self, state, epsilon=0.001):
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()