from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

import numpy as np

class Hubbard():
    """ The Hamiltonian of Hubbard Model
    """

    def __init__(self, Lx, Ly, t, U, num_orbitals = 7):
        """ Init Hubbard model
         
        @param Lx (int): System x size (one dimensionality)
        @param Ly (int): System y size (one dimensionality)
        @param t (float): NN coupling
        @param U (float): Interaction energy 
        @param num_orbitals (int): The number of spin orbitals
        """
        
        self.Lx = Lx
        self.Ly = Ly
        self.sys_size = Lx * Ly
        self.t = t
        self.U = U
        self.num_orbitals = num_orbitals

        self.hop_list_up = []
        self.hop_list_down = []
        self.inter_list = []

        self.H_up = torch.zeros(self.sys_size, self.sys_size)
        self.H_down = torch.zeros(self.sys_size, self.sys_size)

        # Generate the hopping and interaction pairs
        for i in range(Lx):
            for j in range(Ly):
                idx = i * Lx + j
                idxp = ((i + 1) % Lx) * Lx + j
                idyp = i * Lx + ((j + 1) % Ly)

                idxst = min(idx, idxp)
                idxed = max(idx, idxp)
                idyst = min(idx, idyp)
                idyed = max(idx, idyp)
                self.hop_list_up.append((idxst, idxed))
                self.hop_list_up.append((idyst, idyed))
                self.hop_list_down.append((idxst + self.sys_size, idxed + self.sys_size))
                self.hop_list_down.append((idyst + self.sys_size, idyed + self.sys_size))
                
                self.H_up[idx, idxp] = -t; self.H_up[idx, idyp] = -t
                self.H_up[idxp, idx] = -t; self.H_up[idyp, idx] = -t
                self.H_down[idx, idxp] = -t; self.H_down[idx, idyp] = -t
                self.H_down[idxp, idx] = -t; self.H_down[idyp, idx] = -t
                self.inter_list.append((idx, idx + self.sys_size))
                eigvalues_up, eigvectors_up = torch.linalg.eigh(self.H_up)
                self.states_up = eigvectors_up.to(torch.float64)
                eigvalues_down, eigvectors_down = torch.linalg.eigh(self.H_down)
                self.states_down = eigvectors_down.to(torch.float64) 
        self.hop_list = self.hop_list_up + self.hop_list_down

    def local_energy(self, r_list: torch.Tensor, position: torch.Tensor, network: nn.Module, prob_list: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Take a list of configurations

        @param r_list (Tensor): The configuration tensor of size (b, num_orbitals * sys_size)
        @param position (Tensor): The position tensor of size (b, num_orbitals, sys_size)
        @param network (nn.Module): The distribution network
        @param train (bool): Whether to train the network
        @param prob_list (Tensor): The probability list of the configurations

        @returns energy (float): The local energy of the configuration
        """
        device = network.device
        N = self.Lx * self.Ly
        r_list = r_list.to(device)
        N_samples = position.size(0)
        position = position.to(device)

        ## Compute <config|psi>
        over_config_psi = network(r_list,position) # (b,)

        ## Compute <config H|psi>
        N_hop = len(self.hop_list)
        conp_psi = torch.zeros(N_hop, N_samples, dtype=torch.float64, device=device)
        phase = torch.zeros(N_hop, N_samples, dtype=torch.float64, device=device)
        for i, hop in enumerate(self.hop_list):
            stid, edid = hop
            ### Determine whether count the config' or not
            if stid == edid:
                does_count = (r_list[:, stid] > 0) & (r_list[:, edid] > 0)
            else:
                does_count = (r_list[:, stid] > 0) ^ (r_list[:, edid] > 0)
            count_id = torch.nonzero(does_count).squeeze(1)
            r_hop = r_list[count_id]
            position_hop = position[count_id]
            ### Flip the configuration 
            r_hop[:, hop] *= -1
            is_hop0 = position_hop == stid
            is_hop1 = position_hop == edid
            nothop = ~is_hop0 & ~is_hop1
            position_hop = nothop * position_hop + is_hop0 * edid + is_hop1 * stid

            ### Compute the overlap <config'|psi>
            over_hop = network(r_hop, position_hop) # (b,)
            conp_psi[i, count_id] = over_hop
            ### Determine the phase           
            count_positive = torch.sum(((r_hop[:, stid+1:edid]+1)/2), dim=1)
            phase[i, count_id] = 1.0 - 2.0 * (count_positive % 2)
        ## Compute the free local energy
        over_free = torch.sum(conp_psi * phase, 0)
        e_free = -self.t * over_free / over_config_psi
        
        ## Compute the interacting energy
        r_up_list = (r_list[:, :N]>0).float()      # (b, sys_size)
        r_down_list = (r_list[:, N:]>0).float()    # (b, sys_size)
        num_inter = torch.sum(r_up_list * r_down_list, 1) # (b,)
        e_inter = num_inter * self.U                      # (b,)
        ## Compute the total local energy
        e_loc = e_free + e_inter                          # (b,)

        # Here, we need to consider the derivative of psi
        # Therefore, the loss is similar to that in reinforcement learning
        log_psi = torch.log(torch.abs(over_config_psi))
        mean_loce_logpsi = torch.mean(e_loc.detach() * log_psi)
        loce_mean_logpsi = torch.mean(e_loc).detach() * torch.mean(log_psi)

        if prob_list is None:
            E_mean = torch.mean(e_loc)
        else:
            E_mean = torch.sum(e_loc * prob_list)
        return 2*(mean_loce_logpsi - loce_mean_logpsi), E_mean, torch.var(e_loc)
        ## Don't need to add variance into the loss
        #return torch.mean(e_loc) + torch.var(e_loc)
    def local_energy_old(self, r_list: torch.Tensor, network: nn.Module, prob_list: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Take a list of configurations

        @param r_list (Tensor): The configuration tensor of size (b, num_orbitals * sys_size)
        @param position (Tensor): The position tensor of size (b, num_orbitals, sys_size)
        @param network (nn.Module): The distribution network
        @param train (bool): Whether to train the network
        @param prob_list (Tensor): The probability list of the configurations

        @returns energy (float): The local energy of the configuration
        """
        r_list = r_list.to(network.device)
        r_list = r_list.to(torch.float64)
        ## Compute <config|psi>

        ### !!!!Caution: The position should be correct!
        over_config_psi = network([r_list, torch.rand(1,2)]) # (b,)

        ## Compute <config H|psi>
        N_hop = len(self.hop_list)
        over_config_psi = over_config_psi.repeat(N_hop, 1)

        conp_psi = torch.zeros(N_hop, r_list.size(0), dtype=torch.float64, device=r_list.device)
        phase = torch.zeros(N_hop, r_list.size(0), dtype=torch.float64, device=r_list.device)
        for i, hop in enumerate(self.hop_list):
            stid, edid = hop
            ### Determine whether count the config' or not
            if stid == hop[1]:
                does_count = (r_list[:, stid] > 0) & (r_list[:, edid] > 0)
            else:
                does_count = (r_list[:, stid] > 0) ^ (r_list[:, edid] > 0)
                #does_count = (r_list[:, hop[0]] > 0) & (r_list[:, hop[1]] < 0)
            count_id = torch.nonzero(does_count)
            if count_id.size(0) == 0:
                continue
            else:
                count_id = count_id.squeeze(1)
            r_hop = r_list[count_id].clone()
            ### Flip the configuration 
            r_hop[:, stid] = -r_hop[:, stid]
            r_hop[:, edid] = -r_hop[:, edid]
            ### Compute the overlap <config'|psi>
            over_hop = network([r_hop, torch.rand(1,2)]) # (b,)
            conp_psi[i, count_id] = over_hop
            ### Determine the phase
            #count_positive = torch.sum(r_hop[:, stid+1:edid] > 0, dim=1).to(torch.float64)
            #phase[i, count_id] = (-1) ** count_positive
            count_positive = torch.sum(((r_hop[:, stid+1:edid]+1)/2), dim=1)
            phase[i, count_id] = 1.0 - 2.0 * (count_positive % 2)
        ## Compute the free local energy
        if prob_list is None:
            prob_list = torch.ones(r_list.size(0), dtype=torch.float64, device=r_list.device)/r_list.size(0)
        over_all = conp_psi * phase / over_config_psi * prob_list.repeat(N_hop, 1)
        over_free = torch.sum(over_all, 1)

        return -self.t * torch.sum(over_free)

    
    def nk_hoplist(self, nk_list: list) -> Tuple[list, list]:
        """ Generate the hop list for momentum particles

        @param nk_list (list): [[kx_1, ky_1, up(down)], [kx_2, ky_2, up(down)], ...], one need to specify the kx,ky and the spin
        
        @return nkhoplist (list(list)): [[[c^1_{1, dagger}, c^1_{1}], [c^1_{2, dagger}, c^1_{2}], ...]  -> nk_1
                                         [[c^2_{1, dagger}, c^2_{1}], [c^2_{2, dagger}, c^2_{2}], ...]  -> nk_2 
                                         ... -> nk_n   ], the list of hop_list for each nk
        @return nkamplist (list(list)): [[amp^1_1, amp^1_2, ...] -> nk_1
                                         [amp^2_1, amp^2_2, ...] -> nk_2
                                         ......
                                         [amp^n_1, amp^n_2, ...] -> nk_n   ]
        """
        nkhoplist = []
        nkamplist = []

        Lx = self.Lx
        Ly = self.Ly
        # 创建网格，生成所有的 (m1, n1) 和 (m2, n2) 对
        m1, n1 = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij')
        m1 = m1.flatten()
        n1 = n1.flatten()

        m2, n2 = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing='ij')
        m2 = m2.flatten()
        n2 = n2.flatten()

        for kx, ky, spin_dir in nk_list:
            # 计算 hopping operators
            c_dagger_indices = m1 * Lx + n1
            c_indices = m2 * Ly + n2

            if spin_dir:
                # 如果是spin up，则调整 indices
                c_dagger_indices += Lx * Ly
                c_indices += Lx * Ly

            hoplist = np.column_stack((np.repeat(c_dagger_indices, Lx*Ly), np.tile(c_indices, Lx*Ly)))

            # 计算 amplitudes
            amplitude = np.exp(1j * kx * (m1[:, None] - m2) + 1j * ky * (n1[:, None] - n2)) /Lx/Ly
            amplist = amplitude.flatten()

            nkhoplist.append(hoplist)
            nkamplist.append(amplist)
        
        return nkhoplist, nkamplist

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


        
