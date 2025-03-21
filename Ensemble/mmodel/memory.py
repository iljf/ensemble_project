# -*- coding: utf-8 -*-
from __future__ import division
from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal', 'mask', 'Energy'))
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False, torch.zeros(5, dtype=torch.float32), torch.zeros(5, dtype=torch.float32))

#mask = torch.bernoulli(torch.Tensor([ber_mean]*num_ensemble))
# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
        self.data = np.array([None] * size)  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

class ReplayMemory():
    def __init__(self, args, capacity, beta_mean, num_ensemble):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.s = args.gamma
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.beta_mean = beta_mean
        self.num_ensemble = num_ensemble
        self.pre_sample = 1000
        self.t = 0  # Internal episode timestep counter
        self.transitions = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal, Energy):
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
        mask = torch.bernoulli(torch.Tensor([self.beta_mean]*self.num_ensemble))
        self.transitions.append(Transition(self.t, state, action, reward, not terminal, mask, Energy), self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):
        transition = np.array([None] * (self.history + self.n))
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[t + 1].timestep == 0:
                transition[t] = blank_trans  # If future frame has timestep 0
            else:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
            else:
                transition[t] = blank_trans  # If prev (next) frame is terminal
        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.transitions.find(sample)  # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.capacity > self.n and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)
        # Create un-discretised state and nth next state
        state = torch.stack([trans.state for trans in transition[:self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
        next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
        # Discrete action to be used as index
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)
        
        mask = transition[self.history - 1].mask.to(dtype=torch.uint8, device=self.device)
        reliability = transition[self.history -1].Energy
        
        return prob, idx, tree_idx, state, action, R, next_state, nonterminal, mask, reliability

    def sample(self, batch_size, temperature):
        exp = [[] for _ in range(self.num_ensemble)]
        agent_exp = []
        while True:
            p_total = self.transitions.total()
            segment = p_total / self.pre_sample
            batch = [self._get_sample_from_segment(segment, i) for i in range(self.pre_sample)]

            probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals, masks, reliability = zip(*batch)

            reliability = torch.stack([r if isinstance(r, torch.Tensor) else torch.tensor(r) for r in reliability])

            s_probs = torch.softmax(-reliability / temperature, dim=1)
            agent_assignments = [torch.multinomial(s_probs[i], 1).item() for i in range(self.pre_sample)]

            for i, en_index in enumerate(agent_assignments):
                exp[en_index].append(batch[i])

            satisfied = True
            for en_index in range(self.num_ensemble): # ordering samples with priority up to 32
                exp[en_index] = sorted(exp[en_index], key=lambda x: x[0], reverse=True)[:batch_size]
                if len(exp[en_index]) < batch_size:
                    satisfied = False

            if satisfied:
                break

        # Prepare outputs for the specified agent_id
        for en_index in range(self.num_ensemble):
            model_batch = exp[en_index]
            probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals, masks, _ = zip(*model_batch)
            states, next_states = torch.stack(states), torch.stack(next_states)
            actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
            probs = np.array(probs, dtype=np.float32) / p_total
            capacity = self.capacity if self.transitions.full else self.transitions.index
            weights = (capacity * probs) ** -self.priority_weight
            weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)

            masks = torch.cat(masks, dim=0)
            masks = masks.reshape(-1, self.num_ensemble)

            agent_exp.append((probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals, masks, weights))
        return agent_exp

    def get_transition_Energy(self, idx):
        transition = self.transitions.get(idx)
        return transition.Energy

    def update_transition(self, idx, updated_mse):
        self.transitions.data[idx % self.capacity] = self.transitions.data[idx % self.capacity]._replace(Energy=updated_mse)


    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[t] = blank_trans.state  # If future frame has timestep 0
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(255)  # Agent will turn into batch
        self.current_idx += 1
        return state

    next = __next__  # Alias __next__ for Python 2 compatibility