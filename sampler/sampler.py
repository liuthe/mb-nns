# Description: The sampler module for the Monte Carlo simulation
import torch
import torch.nn as nn

def help_flip(position: torch.Tensor, Nocu: int, Nswap: int) -> torch.Tensor:
    """ Flip the position matrix

    @param position (tensor): The positions of the config of size (LL, num_chains),
    the first Nup rows record the ids of the occupied sites
    @param Nup (int): The number of occupied sites
    @param Nswap (int): The number of sites to be swapped

    @returns new_position (tensor): The positions of the updated config
    """
    LL = position.size(0)
    num_chains = position.size(1)
    #### Select the ids of sites for swap
    id_ocu_swap = torch.stack([torch.randperm(Nocu)[:Nswap] for _ in range(num_chains)])             #(num_chains, Nswap)
    id_uno_swap = torch.stack([torch.randperm(LL - Nocu)[:Nswap] for _ in range(num_chains)]) + Nocu #(num_chains, LL-Nswap)        
    ele_ocu_swap = position[id_ocu_swap, torch.arange(num_chains).unsqueeze(1)]
    ele_uno_swap = position[id_uno_swap, torch.arange(num_chains).unsqueeze(1)]
    #### Swap the elements
    new_position = position.clone()
    new_position[id_ocu_swap, torch.arange(num_chains).unsqueeze(1)] = ele_uno_swap
    new_position[id_uno_swap, torch.arange(num_chains).unsqueeze(1)] = ele_ocu_swap
    #new_position = new_position.to(position.device)
    return new_position

def help_position2config(positionup: torch.Tensor, Nup: int, positiondown: torch.Tensor, Ndown: int) -> torch.Tensor:
    """ Convert the position matrix to the config tensor
    
    @param positionup (tensor): The positions of the config of size (LL, num_chains),
    the first Nup rows record the ids of the occupied sites
    @param Nup (int): The number of occupied sites for spin up
    @param positiondown (tensor): The positions of the config of size (LL, num_chains),
    @param Ndown (int): The number of occupied sites for spin down
    
    @returns config (tensor): The config tensor of size (num_chains, 2*LL)"""
    LL = positionup.size(0)
    num_chains = positionup.size(1)
    config = torch.full((2*LL, num_chains), -1, dtype=torch.float, device=positionup.device)
    id_up_ocu = positionup[:Nup, :]
    ele_up_ocu = torch.ones((Nup, num_chains), dtype=config.dtype, device=positionup.device)
    id_down_ocu = positiondown[:Ndown, :] + LL
    ele_down_ocu = torch.ones((Ndown, num_chains), dtype=config.dtype, device=positionup.device)
    config.scatter_(0, id_up_ocu, ele_up_ocu)
    config.scatter_(0, id_down_ocu, ele_down_ocu)
    return config.T

def generate_sample(model: nn.Module, num_samples: int, num_chains: int = 1, drop_rate: float = 0.3, sweep_size: int = None) -> torch.Tensor:
    """ Generate samples from the model.

    @param model (NNB): The neural network model
    @param num_samples (int): The number of samples to generate
    @param num_chains (int): The number of Monte Carlo chains

    @returns samples (Tensor): A tensor of shape (num_samples * num_chains, 2L^2)
    """
    
    # Generate samples
    LL = model.input_size
    Nup = model.num_fillings[0]
    Ndown = model.num_fillings[1]
    drop_samples = round(num_samples * drop_rate)
    num_samples = num_samples + drop_samples
    samples = torch.zeros(num_samples, num_chains, 2 * LL, dtype=torch.int)
    prob_list = torch.zeros(num_samples, num_chains, dtype=torch.complex64)
    # The first num_fillings[0]/[1] represents the filled sites for spin up/down
    positions_up = torch.stack([torch.randperm(LL) for _ in range(num_chains)], dim=1)    #(LL, num_chains)
    positions_down = torch.stack([torch.randperm(LL) for _ in range(num_chains)], dim=1)  #(LL, num_chains)
    positions_up = positions_up.to(model.device)
    positions_down = positions_down.to(model.device)
    prob_old = torch.ones(num_chains)  #(num_chains)

    if sweep_size is None:
        sweep_size = LL * 2

    for i in range(num_samples * sweep_size):
        # Metropolis-Hastings
        Nswap = 1
        new_positions_up = help_flip(positions_up, Nup, Nswap)
        new_positions_down = help_flip(positions_down, Ndown, Nswap)
        # Create the new configuration
        config = help_position2config(new_positions_up, Nup, new_positions_down, Ndown)
        config = config.to(model.device)         
        # Calculate the acceptance probability
        prob_amp_new = model.forward(config)
        prob_new = torch.abs(prob_amp_new) ** 2
        if i == 0:
            prob_old = prob_new
            samples[i,:,:] = config
            prob_list[i] = prob_old
            continue
        else:
            acceptance = torch.rand(num_chains, device=model.device) < prob_new / prob_old
            positions_up = torch.where(acceptance.unsqueeze(0), new_positions_up, positions_up)
            positions_down = torch.where(acceptance.unsqueeze(0), new_positions_down, positions_down)
            
            prob_old = torch.where(acceptance.unsqueeze(0), prob_new, prob_old).squeeze()
            config = help_position2config(positions_up, Nup, positions_down, Ndown)
            config = config.to(model.device)
            
            if i % sweep_size == 0:
                samples[i//sweep_size,:,:] = config
                prob_list[i//sweep_size] = prob_old
            #samples[i,:,:] = config
            #prob_list[i] = prob_old
    samples = samples[drop_samples:]
    samples = samples.view(-1, 2 * LL)
    prob_list = prob_list.view(-1)
    return samples, prob_list
