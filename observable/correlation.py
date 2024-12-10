import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from collections import namedtuple

def correlation(hop_list: list, r_list: torch.Tensor, network: nn.Module, prob_list: torch.Tensor = None) -> torch.Tensor:
        """ Compute <c_i^\dagger c_j + c_j^\dagger c_i> or <n_i> correlation functions

        @param hop_list (list): The list of correlation operators (N_hop)
        @param r_list (Tensor): The configuration tensor of size (b, num_orbitals * sys_size)
        @param network (nn.Module): The distribution network

        @returns energy (float): The local energy of the configuration
        """
        r_list = r_list.to(network.device)
        r_list = r_list.float()
        ## Compute <config|psi>
        over_config_psi = network(r_list) # (b,)

        ## Compute <config H|psi>
        N_hop = len(hop_list)
        over_config_psi = over_config_psi.repeat(N_hop, 1)

        conp_psi = torch.zeros(N_hop, r_list.size(0), dtype=torch.float64, device=r_list.device)
        phase = torch.zeros(N_hop, r_list.size(0), dtype=torch.float64, device=r_list.device)
        for i, hop in enumerate(hop_list):
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
            count_positive = torch.sum(r_hop[:, stid+1:edid] > 0, dim=1).to(torch.float64)
            phase[i, count_id] = (-1) ** count_positive
        ## Compute the free local energy
        if prob_list is None:
            prob_list = torch.ones(r_list.size(0), dtype=torch.float64, device=r_list.device)/r_list.size(0)
        over_all = conp_psi * phase / over_config_psi * prob_list.repeat(N_hop, 1)
        over_free = torch.sum(over_all, 1)
                           
        ## Compute the total local energy
        return over_free

def correlation_nk(k_list: list, r_list: torch.Tensor, network: nn.Module, prob_list: torch.Tensor = None) -> torch.Tensor:
     
     return None

