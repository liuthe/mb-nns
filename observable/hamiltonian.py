from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

import math

class Hubbard():
    """ The Hamiltonian of Hubbard Model
    """

    def __init__(self, L, t, U, num_orbitals = 7):
        """ Init Hubbard model
         
        @param L (int): System size (one dimensionality)
        @param t (float): NN coupling
        @param U (float): Interaction energy 
        @param num_orbitals (int): The number of spin orbitals
        """
        
        self.L = L
        self.sys_size = L * L
        self.t = t
        self.U = U
        self.num_orbitals = num_orbitals

        self.hop_list_up = []
        self.hop_list_down = []
        self.inter_list = []

        self.H_up = torch.zeros(self.sys_size, self.sys_size)
        self.H_down = torch.zeros(self.sys_size, self.sys_size)

        # Generate the hopping and interaction pairs
        for i in range(L):
            for j in range(L):
                idx = i * L + j
                idxp = ((i + 1) % L) * L + j
                idyp = i * L + ((j + 1) % L)
                self.hop_list_up.append((idx, idxp))
                self.hop_list_up.append((idx, idyp))
                self.hop_list_down.append((idx + self.sys_size, idxp + self.sys_size))
                self.hop_list_down.append((idx + self.sys_size, idyp + self.sys_size))
                
                self.H_up[idx, idxp] = -t; self.H_up[idx, idyp] = -t
                self.H_up[idxp, idx] = -t; self.H_up[idyp, idx] = -t
                self.H_down[idx, idxp] = -t; self.H_down[idx, idyp] = -t
                self.H_down[idxp, idx] = -t; self.H_down[idyp, idx] = -t
                self.inter_list.append((idx, idx + self.sys_size))
                eigvalues_up, eigvectors_up = torch.linalg.eigh(self.H_up)
                self.states_up = eigvectors_up
                eigvalues_down, eigvectors_down = torch.linalg.eigh(self.H_down)
                self.states_down = eigvectors_down
    def local_energy_up(self, r_list: torch.Tensor, network, hop_list: list = None) -> float:
        """ Take a list of configurations, compute the up spin energy

        @param r_list (Tensor): The configuration tensor of size (b, sys_size)
        @param network (Any): The distribution network

        @returns energy (float): The local energy of the configuration
        """
        N = self.L * self.L
        Nup = torch.sum(r_list[0,:]>0)
        #assert r_list.size(1) == 2 * N
        ## Compute <config|psi>
        r_list = r_list.float()
        over_config_psi = network(self.states_up[:,:Nup], r_list) # (b,)

        ## Compute <config H|psi>
        if hop_list is None:
            combined_hop = self.hop_list_up 
        else:
            combined_hop = hop_list
        N_hop = len(combined_hop)
        conp_psi = torch.zeros(N_hop, r_list.size(0), dtype=torch.float, device=r_list.device)
        phase = torch.zeros(N_hop, r_list.size(0), dtype=torch.float, device=r_list.device)
        for i, hop in enumerate(combined_hop):
            r_hop = r_list.clone()
            ### Determine whether count the config' or not
            if hop[0] == hop[1]:
                does_count = (r_hop[:, hop[0]] > 0) & (r_hop[:, hop[1]] > 0)
            else:
                does_count = (r_hop[:, hop[0]] > 0) ^ (r_hop[:, hop[1]] > 0)
            count_id = torch.nonzero(does_count)
            if count_id.size(0) == 0:
                continue
            else:
                count_id = count_id.squeeze(1)
            r_hop = r_hop[count_id]
            ### Flip the configuration 
            r_hop[:, hop[0]] = -r_hop[:, hop[0]]
            r_hop[:, hop[1]] = -r_hop[:, hop[1]]
            ### Compute the overlap <config'|psi>
            over_hop = network(self.states_up[:,:Nup], r_hop) # (b,)
            conp_psi[i, count_id] = over_hop
            ### Determine the phase
            stid = min(hop[0], hop[1])
            edid = max(hop[0], hop[1])
            count_positive = torch.sum(r_hop[:, stid+1:edid] > 0, dim=1).float()
            phase[i, count_id] = (-1) ** count_positive
        ## Compute the free local energy
        e_free = -self.t * torch.sum(conp_psi * phase, 0) / over_config_psi                  # (b,)
        ## Compute the total local energy
        e_loc = e_free
        return e_loc   


    def local_energy(self, r_list: torch.Tensor, network: nn.Module, train: bool = False) -> float:
        """ Take a list of configurations

        @param r_list (Tensor): The configuration tensor of size (b, num_orbitals * sys_size)
        @param network (nn.Module): The distribution network

        @returns energy (float): The local energy of the configuration
        """
        N = self.L * self.L
        r_list = r_list.to(network.device)
        r_list = r_list.float()
        ## Compute <config|psi>
        over_config_psi = network(r_list) # (b,)

        ## Compute <config H|psi>
        combined_hop = self.hop_list_up + self.hop_list_down
        N_hop = len(combined_hop)
        conp_psi = torch.zeros(N_hop, r_list.size(0), dtype=torch.float, device=r_list.device)
        phase = torch.zeros(N_hop, r_list.size(0), dtype=torch.float, device=r_list.device)
        for i, hop in enumerate(combined_hop):
            ### Determine whether count the config' or not
            if hop[0] == hop[1]:
                does_count = (r_list[:, hop[0]] > 0) & (r_list[:, hop[1]] > 0)
            else:
                does_count = (r_list[:, hop[0]] > 0) ^ (r_list[:, hop[1]] > 0)
            count_id = torch.nonzero(does_count)
            if count_id.size(0) == 0:
                continue
            else:
                count_id = count_id.squeeze(1)
            r_hop = r_list[count_id].clone()
            ### Flip the configuration 
            r_hop[:, hop[0]] = -r_hop[:, hop[0]]
            r_hop[:, hop[1]] = -r_hop[:, hop[1]]
            ### Compute the overlap <config'|psi>
            over_hop = network(r_hop) # (b,)
            conp_psi[i, count_id] = over_hop
            ### Determine the phase
            stid = min(hop[0], hop[1])
            edid = max(hop[0], hop[1])
            count_positive = torch.sum(r_hop[:, stid+1:edid] > 0, dim=1).float()
            phase[i, count_id] = (-1) ** count_positive
        ## Compute the free local energy
        over_free = torch.sum(conp_psi * phase, 0)
        nonzero_id = torch.nonzero(torch.abs(over_config_psi))[0]
        #e_free = -self.t * torch.sum(conp_psi * phase, 0) / over_config_psi
        e_free = -self.t * over_free[nonzero_id] / over_config_psi[nonzero_id]
        ## Compute the interacting energy
        r_up_list = (r_list[:, :N]>0).float()      # (b, sys_size)
        r_down_list = (r_list[:, N:]>0).float()    # (b, sys_size)
        num_inter = torch.sum(r_up_list * r_down_list, 1) # (b,)
        e_inter = num_inter[nonzero_id] * self.U                      # (b,)
        ## Compute the total local energy
        e_loc = e_free + e_inter
        if train:
            return torch.mean(e_loc) + torch.var(e_loc)
        else:
            return torch.mean(e_loc)

def config2state(config):
    """ Convert a configuration to a state
    
    @param config (Tensor)): The configuration of the system, a bool tensor of size (sys_size * num_orbitals)

    @returns state (Tensor): The state of the system, a float tensor of size (sys_size * num_orbitals, num_fillings)
    """

    true_indices = torch.nonzero(config).squeeze()
    r_state = torch.zeros(len(config), len(true_indices))
    for i, idx in enumerate(true_indices):
        r_state[idx, i] = 1
    return r_state

def batch2state(batch):
    """ Convert a batch of configurations to a batch of states

    @param batch (Tensor): The batch of configurations of the system, a bool tensor of size (b, sys_size)

    @returns state (Tensor): The batch of states of the system, a float tensor of size (b, sys_size, num_fillings)
    """
    device = batch.device
    true_indices = torch.nonzero(batch).to(device)
    num_fillings = len(true_indices)
    true_fillings = num_fillings//batch.size(0)
    state = torch.zeros(batch.size(0), batch.size(1), true_fillings, device=device)
    last_dim_indices = torch.arange(num_fillings) % true_fillings 
    state[true_indices[:, 0], true_indices[:, 1], last_dim_indices] = 1
    return state


        
