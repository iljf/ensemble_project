# -*- coding: utf-8 -*-
from __future__ import division
from collections import namedtuple, defaultdict
import numpy as np
import torch
import random
import sys

# Segment tree data structure where parent node values are sum/max of children node values

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'nonterminal'))

class GWRNode:
    def __init__(self, state):
        self.state = state
        self.edges = defaultdict(list)
        self.habituation = 1.0

class GWRMemory:
    def __init__(self, args):
        self.nodes = []
        self.history = args.history_length
        self.activation_threshold = args.activation_threshold
        self.habituation_threshold = args.habituation_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _find_bmu(self, state):
        state = state.to(dtype=torch.float32)
        min_dist = float('inf')
        bmu = None
        for node in self.nodes:
            node_state = node.state.to(dtype=torch.float32)
            dist = torch.norm(state - node_state)
            if dist < min_dist:
                min_dist = dist
                bmu = node
        return bmu, min_dist

    def append(self, state, action, reward, next_state, nonterminal):
        state = state[-4:] if state.shape[0] > 4 else state
        next_state = next_state[-4:] if next_state.shape[0] > 4 else next_state

        state = state.mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))
        next_state = next_state.mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))

        if not self.nodes:
            new_node = GWRNode(state)
            new_node.edges[action] = [(reward, next_state, nonterminal)]
            self.nodes.append(new_node)
        else:
            bmu, dist = self._find_bmu(state)
            if dist < self.activation_threshold and bmu.habituation > self.habituation_threshold:
                bmu.edges.setdefault(action, []).append((reward, next_state, nonterminal))
            else:
                new_node = GWRNode(state)
                new_node.edges[action] = [(reward, next_state, nonterminal)]
                self.nodes.append(new_node)

    def sample(self, batch_size):
        transitions = []
        for _ in range(batch_size):
            node = random.choice(self.nodes)
            action = random.choice(list(node.edges.keys()))
            edges = node.edges[action]
            reward, next_state, nonterminal = random.choice(edges)

            state = node.state[-4:] if node.state.shape[0] > 4 else node.state
            next_state = next_state[-4:] if next_state.shape[0] > 4 else next_state

            transitions.append(Transition(state, action, reward, next_state, nonterminal))

        batch = Transition(*zip(*transitions))
        states = torch.stack(batch.state).to(self.device).float() / 255.0
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.stack(batch.next_state).to(self.device).float() / 255.0
        nonterminals = torch.tensor(batch.nonterminal, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, nonterminals

    def memory_size(self):
        total_nodes = len(self.nodes)
        total_edges = 0

        for node in self.nodes:
            for action, edges in node.edges.items():
                total_edges += len(edges)

        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'total_size': total_nodes + total_edges
        }

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.nodes):
            raise StopIteration

        node = self.nodes[self.current_idx]
        state = node.state

        self.current_idx += 1

        return state.float() / 255.0



