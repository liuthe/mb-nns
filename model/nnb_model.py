from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

import numpy as np
from model.hamiltonian import Hubbard, config2state

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class NNB_single_spin(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NNB_single_spin, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self,x):
        return self.layers(x)

class NNB(nn.Module):
    """ Simple Neural Network Backflow Model. (Luo, et al. 2019)
    """

    def __init__(self, sys_size, hidden_size, num_fillings):
        """ Init NNB Model.

        @param sys_size (int): System size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param num_fillings (list of int): The number of fillings for spin up and spin down
        """
        super(NNB, self).__init__()

        self.input_size = sys_size
        self.hidden_size = hidden_size
        self.num_fillings = num_fillings
        self.output_size_up = num_fillings[0] * sys_size
        self.output_size_down = num_fillings[1] * sys_size

        # default values
        self.layer_up = NNB_single_spin(self.input_size, self.hidden_size, self.output_size_up)
        self.layer_down = NNB_single_spin(self.input_size, self.hidden_size, self.output_size_down)

        # Initialize device property
        self._device = torch.device("cpu")  # Default to CPU

    def forward(self, config_fermion):
        """ Take a mini-batch of configurations of fermions, compute the probabilities of each distributiob.

        @param config_fermion (Tensor): A batch of configurations of fermions, shape (b, 2 * sys_size)

        @returns probability (Tensor): a variable/tensor of shape (b, 2 * sys_size * num_fillings) representing the
                                    probability distribution of each fermion configuration. Here b = batch size.
        """
        # Divide the input into up and down spins
        config_fermion_up = config_fermion[:, :self.input_size] # (b, input_size)
        config_fermion_down = config_fermion[:, self.input_size:] # (b, input_size)
        
        # Apply the first linear layer
        out_up = self.layer_up(config_fermion_up) # (b, output_size_up)
        out_down = self.layer_down(config_fermion_down) # (b, output_size_down)
  
        # Reshape the output
        out_up = out_up.view(-1, self.input_size, self.num_fillings[0]) # (b, input_size, num_fillings_up)
        out_down = out_down.view(-1, self.input_size, self.num_fillings[1]) # (b, input_size, num_fillings_down)

        return out_up, out_down

    def generate_sample(self, num_samples):
        """ Generate samples from the model.

        @param num_samples (int): The number of samples to generate

        @returns samples (Tensor): A tensor of shape (num_samples, 2 * sys_size)
        """
        # Generate samples
        samples = torch.zeros(num_samples, 2 * self.input_size, dtype=torch.int)
        prob_list = torch.zeros(num_samples, dtype=torch.complex64)
        # The first num_fillings[0] represents the filled sites for spin up
        positions_up = torch.randperm(self.input_size)
        # The next num_fillings[1] represents the filled sites for spin down
        positions_down = torch.randperm(self.input_size)
        prob_old = 1
        for i in range(num_samples):
            # Metropolis-Hastings
            ## Randomly select n out of L occupied sites and n unoccupied sites to swap.
            ## As the number of up spins and down spins are converved separately,
            ## we swap the occupied and unoccupied sites for each spin separately.
            n = 1
            ### For spin up
            indices_up_occupied = torch.randperm(self.num_fillings[0])[:n]
            indices_up_unoccupied = torch.randperm(self.input_size - self.num_fillings[0])[:n] + self.num_fillings[0]           
            elements_up_occupied = positions_up[indices_up_occupied]
            elements_up_unoccupied = positions_up[indices_up_unoccupied]
            ### For spin down
            indices_down_occupied = torch.randperm(self.num_fillings[1])[:n]
            indices_down_unoccupied = torch.randperm(self.input_size - self.num_fillings[1])[:n] + self.num_fillings[1]           
            elements_down_occupied = positions_down[indices_down_occupied]
            elements_down_unoccupied = positions_down[indices_down_unoccupied]
            ## Clone the occupied and unoccupied sites, and swap the elements
            new_positions_up = positions_up.clone()
            new_positions_up[indices_up_occupied] = elements_up_unoccupied
            new_positions_up[indices_up_unoccupied] = elements_up_occupied

            new_positions_down = positions_down.clone()
            new_positions_down[indices_down_occupied] = elements_down_unoccupied
            new_positions_down[indices_down_unoccupied] = elements_down_occupied
            # Create the new configuration
            config = torch.ones(2 * self.input_size, dtype=torch.float)
            config[new_positions_up[self.num_fillings[0]:]] = -1
            config[new_positions_down[self.num_fillings[1]:] + self.input_size] = -1

            config = config.to(self._device)
            out_config_up, out_config_down = self.forward(config.unsqueeze(0))
            
            # Calculate the acceptance probability
            overlap_up = torch.det(out_config_up[:, new_positions_up[:self.num_fillings[0]], :])
            overlap_down = torch.det(out_config_down[:, new_positions_down[:self.num_fillings[1]], :])
            overlap = torch.abs(overlap_up * overlap_down) ** 2
            prob_new = overlap[0].item()
            if i == 0:
                prob_old = prob_new
            # Accept or reject the new configuration
            if torch.rand(1).item() < prob_new / prob_old:
                new_positions_up = positions_up
                new_positions_down = positions_down
                prob_old = prob_new
                samples[i, :] = config
                prob_list[i] = (overlap_up * overlap_down)[0].item()
            else:
                samples[i, :] = samples[i-1, :]
                prob_list[i] = prob_list[i-1]

        return samples, prob_list

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self._device

    def to(self, device):
        """ Move the model to the specified device.
        """
        self._device = device
        return super(NNB, self).to(device)

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NNB(**args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(sys_size=self.input_size, hidden_size=self.hidden_size,
                         num_fillings=self.num_fillings),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)