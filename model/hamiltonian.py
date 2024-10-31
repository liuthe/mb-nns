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
                
                self.H_up[idx, idxp] = t; self.H_up[idx, idyp] = t
                self.H_up[idxp, idx] = t; self.H_up[idyp, idx] = t
                self.H_down[idx, idxp] = t; self.H_down[idx, idyp] = t
                self.H_down[idxp, idx] = t; self.H_down[idyp, idx] = t
                self.inter_list.append((idx, idx + self.sys_size))
    
    def energy_local_single(self, r, psi_list, p_list = torch.tensor([1.0])):
        """ Take a single configuration 

        @param r (Tensor): The configuration tensor of size (num_orbitals * sys_size, )
        @param psi_list (Tensor): The wavefunction tensor of size (num_psi, num_orbitals * sys_size, num_fillings)
        @param p_list (Tensor): The list of weights of each psi of psi_list of size (num_psi, )

        @returns energy (float): The local energy of the configuration
        """
        # Prepare the psi list for multiplication
        L = math.isqrt(self.sys_size)
        if psi_list.dim() == 2:
            psi_list = psi_list.unsqueeze(0)
        psi_list_conj = psi_list.transpose(1, 2).conj()

        # Calculate the c_i^\dagger c_i+1 |r> list
        num_fillings = sum(r).item()
        num_hop = len(self.hop_list)
        rp_list = torch.zeros(num_hop, self.sys_size * self.num_orbitals, num_fillings)

        for id_hop in range(num_hop):
            (id_crea, id_ann) = self.hop_list[id_hop]
            if r[id_crea] == 1 and r[id_ann] == 0:
                rp = r.clone(); rp[id_crea] = 0; rp[id_ann] = 1
                rp_list[id_hop, :, :] = config2state(rp)
            elif r[id_crea] == 0 and r[id_ann] == 1:
                rp = r.clone(); rp[id_crea] = 1; rp[id_ann] = 0
                rp_list[id_hop, :, :] = config2state(rp)

        # Count the number of interactions. The interaction term does not change the |r> state, so we just count the number of interactions
        inter_count = 0
        r_state = config2state(r)
        for (id_crea, id_ann) in self.inter_list:
            inter_count += r[id_crea] * r[id_ann]
        
        # Now compute the total Local energy
        num_psi = psi_list.size(0)
        psi = psi_list_conj.unsqueeze(1).expand(-1, num_hop, -1, -1) # (num_psi, num_hop, num_fillings, num_orbitals * sys_size)
        rp_list = rp_list.unsqueeze(0).expand(num_psi, -1, -1, -1) # (num_psi, num_hop, num_orbitals * sys_size, num_fillings)

        e_hop = -self.t * torch.sum(torch.det(torch.matmul(psi, rp_list)), 1)
        e_inter = self.U * inter_count * torch.det(torch.matmul(psi_list_conj, r_state.unsqueeze(0)))

        e_total = torch.matmul(p_list, e_hop + e_inter).item()
        return e_total
    
    def energy_local(self, r_list, nnb):
        """ Take a list of configurations

        @param r_list (Tensor): The configuration tensor of size (b, num_orbitals * sys_size)
        @param nnb (NNB): The neural network model

        @returns energy (float): The local energy of the configuration
        """
        ## Move the configurations to the device
        r_list = r_list.to(nnb.device)
        self.H_up = self.H_up.to(nnb.device)
        self.H_down = self.H_down.to(nnb.device)
        ## Change the (1,-1,...) to (1,0,...)
        r_list = torch.where(r_list > 0, 1.0, 0.0) # (b, num_orbitals * sys_size)
        ## Construct the state for spin up and down
        r_up_list = r_list[:, :self.sys_size]      # (b, sys_size)
        r_down_list = r_list[:, self.sys_size:]    # (b, sys_size)
        state_up_list = batch2state(r_up_list)     # (b, sys_size, num_fillings)
        state_up_list = state_up_list.transpose(1, 2).conj()     #(b, num_fillings, sys_size)
        state_down_list = batch2state(r_down_list) # (b, sys_size, num_fillings)
        state_down_list = state_down_list.transpose(1, 2).conj() #(b, num_fillings, sys_size)
        out_up, out_down = nnb.forward(2 * r_list - 1)           #((b, num_fillings, sys_size), (b, num_fillings, sys_size))
        ## Now compute the overlap <state|psi> and <statep|psi> = <state H|psi>
        ## These calculations gives the free energy part E_free = sum_state <statep|psi>/<state|psi>
        over_up = torch.det(torch.bmm(state_up_list, out_up))    #(b,)
        statep_up_list = torch.matmul(state_up_list, self.H_up)  #(b, num_fillings, sys_size)
        overp_up = torch.det(torch.bmm(statep_up_list, out_up))  #(b,)
        over_down = torch.det(torch.bmm(state_down_list, out_down))    #(b,)
        statep_down_list = torch.matmul(state_down_list, self.H_down)  #(b, num_fillings, sys_size)
        overp_down = torch.det(torch.bmm(statep_down_list, out_down))  #(b,)

        overp = overp_up * overp_down               # (b,)
        over = over_up * over_down                  # (b,)
        e_free = overp / over                       # (b,)
        ## Now we compute the interacting energy
        num_inter = torch.sum(r_up_list * r_down_list, 1) # (b,)
        e_inter = num_inter * self.U                      # (b,)

        e_total = torch.sum(e_free + e_inter) / r_list.size(0)
        return e_total


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


        
