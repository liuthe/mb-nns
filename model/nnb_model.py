from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

import numpy as np
from observable.hamiltonian import Hubbard, config2state
from sampler.sampler import help_flip, help_position2config

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class RBM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RBM, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, dtype=torch.float64),
        )
    def forward(self,x):
        return self.layers(x)

class NNB(nn.Module):
    """ Simple Neural Network Backflow Model. (Luo, et al. 2019)
    """
    def __init__(self, sys_size: int, num_fillings: list):
        """ Init NNB Model.

        @param sys_size (int): System size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param num_fillings (list of int): The number of fillings for spin up and spin down
        """
        super(NNB, self).__init__()

        self.input_size = sys_size
        self.num_fillings = num_fillings
        self.output_size_up = num_fillings[0] * sys_size
        self.output_size_down = num_fillings[1] * sys_size
        # Initialize device property
        self._device = torch.device("cpu")  # Default to CPU 

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

class spinsep_NNB(NNB):
    """ Neural Network Backflow Model with Spin Separation. (Luo, et al. 2019)"""
    def __init__(self, sys_size: int, hidden_size: int, num_fillings: list):
        super(spinsep_NNB, self).__init__(sys_size, num_fillings)
        self.hidden_size = hidden_size
        # default values
        self.layer_up = RBM(self.input_size, self.hidden_size, self.output_size_up)
        self.layer_down = RBM(self.input_size, self.hidden_size, self.output_size_down)
        self.p_neuron = nn.Parameter(torch.randn(1)) * 0.1

    def forward(self, config_fermion):
        """ Take a mini-batch of configurations of fermions, compute the probabilities of each distributiob.

        @param config_fermion (Tensor): A batch of configurations of fermions, shape (b, 2 * sys_size)

        @returns probability (Tensor): a variable/tensor of shape (b, 2 * sys_size * num_fillings) representing the
                                    probability distribution of each fermion configuration. Here b = batch size.
        """
        config_fermion = config_fermion.to(torch.float64)
        Nsamples = config_fermion.size(0)
        Nup = self.num_fillings[0]
        # test whether the input has the required up and down spins
        Ndown = self.num_fillings[1]
        assert (Nup+Ndown) == torch.sum(config_fermion[0]>0)
        # Divide the input into up and down spins
        config_fermion_up = config_fermion[:, :self.input_size] # (b, input_size)
        config_fermion_down = config_fermion[:, self.input_size:] # (b, input_size)
        
        # Apply the first linear layer and reshape the output
        out_up = self.layer_up(config_fermion_up) # (b, output_size_up)
        out_down = self.layer_down(config_fermion_down) # (b, output_size_down)
        out_up = out_up.view(-1, self.input_size, Nup) # (b, input_size, num_fillings_up)
        out_down = out_down.view(-1, self.input_size, Ndown) # (b, input_size, num_fillings_down)

        # Select the filled states
        filled_up = torch.nonzero(config_fermion_up == 1).reshape(-1, Nup, 2)
        filled_down = torch.nonzero(config_fermion_down == 1).reshape(-1, Ndown, 2)

        matrices_up = out_up[torch.arange(Nsamples).unsqueeze(1), filled_up[:,:,1]]
        matrices_down = out_down[torch.arange(Nsamples).unsqueeze(1), filled_down[:,:,1]]

        det_up = torch.det(matrices_up)
        det_down = torch.det(matrices_down)
        det_all = det_up * det_down
        return det_all
    
    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = spinsep_NNB(**args)
        model.load_state_dict(params['state_dict'])
        return model
    
class free_NNB(NNB):
    """ Neural Network Model with Exact Free Parameters."""
    def __init__(self, sys_size: int, num_fillings: list, h_model: object = None):
        super(free_NNB, self).__init__(sys_size, num_fillings)
        self.h_model = h_model

    def forward(self, config_fermion):
        """ Take a mini-batch of configurations of fermions, compute the probabilities of each distributiob.

        @param config_fermion (Tensor): A batch of configurations of fermions, shape (b, 2 * sys_size)

        @returns probability (Tensor): a variable/tensor of shape (b, 2 * sys_size * num_fillings) representing the
                                    probability distribution of each fermion configuration. Here b = batch size.
        """
        config_fermion = config_fermion.to(torch.float64)
        Nsamples = config_fermion.size(0)
        Nup = self.num_fillings[0]
        # test whether the input has the required up and down spins
        Ndown = self.num_fillings[1]
        assert (Nup+Ndown) == torch.sum(config_fermion[0]>0)
        # Divide the input into up and down spins
        config_fermion_up = config_fermion[:, :self.input_size] # (b, input_size)
        config_fermion_down = config_fermion[:, self.input_size:] # (b, input_size)
        
        # Apply the first linear layer and reshape the output
        ## Use the true ground state to compute the probability, and makes b copies of them
        state_up = self.h_model.states_up[:,:Nup].to(self.device)
        out_up = state_up.unsqueeze(0).expand(Nsamples, -1, -1) # (b, input_size, num_fillings_up)
        state_down = self.h_model.states_down[:,:Ndown].to(self.device)
        out_down = state_down.unsqueeze(0).expand(Nsamples, -1, -1) # (b, input_size, num_fillings_down)
        # Select the filled states
        filled_up = torch.nonzero(config_fermion_up == 1).reshape(-1, Nup, 2)
        filled_down = torch.nonzero(config_fermion_down == 1).reshape(-1, Ndown, 2)

        matrices_up = out_up[torch.arange(Nsamples).unsqueeze(1), filled_up[:,:,1]]
        matrices_down = out_down[torch.arange(Nsamples).unsqueeze(1), filled_down[:,:,1]]

        det_up = torch.det(matrices_up)
        det_down = torch.det(matrices_down)
        det_all = det_up * det_down
        return det_all
    
def spinsum_NNB(NNB):
    """ Neural Network Backflow Model with Spin Separation. (Luo, et al. 2019)"""
    def __init__(self, sys_size: int, hidden_size: int, num_fillings: list):
        super(spinsep_NNB, self).__init__(sys_size, num_fillings)
        self.hidden_size = hidden_size
        # default values
        self.layer_all = RBM(self.input_size * 2, self.hidden_size, self.output_size_up + self.output_size_down)
        self.p_neuron = nn.Parameter(torch.randn(1)) * 0.1

    def forward(self, config_fermion):
        """ Take a mini-batch of configurations of fermions, compute the probabilities of each distributiob.

        @param config_fermion (Tensor): A batch of configurations of fermions, shape (b, 2 * sys_size)

        @returns probability (Tensor): a variable/tensor of shape (b,) representing the
                                    probability distribution of each fermion configuration. Here b = batch size.
        """
        config_fermion = config_fermion.to(torch.float64)
        Nsamples = config_fermion.size(0)
        Nup = self.num_fillings[0]
        # test whether the input has the required up and down spins
        Ndown = self.num_fillings[1]
        assert (Nup+Ndown) == torch.sum(config_fermion[0]>0)
        
        # Apply the first linear layer and reshape the output
        out = self.layer_all(config_fermion) # (b, output_size_up)
        out = out.view(-1, self.input_size * 2, Nup + Ndown) # (b, input_size, num_fillings_up)

        # Select the filled states
        filled = torch.nonzero(config_fermion == 1).reshape(-1, Nup+Ndown, 2)
        matrices = out[torch.arange(Nsamples).unsqueeze(1), filled[:,:,1]]

        det = torch.det(matrices)
        return det
    
    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = spinsum_NNB(**args)
        model.load_state_dict(params['state_dict'])
        return model
    
