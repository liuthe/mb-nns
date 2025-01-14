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
    device = position.device
    LL = position.size(0)
    num_chains = position.size(1)

    help_id = torch.arange(num_chains, device=device).unsqueeze(1)
    #### Select the ids of sites for swap
    id_ocu_swap = torch.stack([torch.randperm(Nocu)[:Nswap] for _ in range(num_chains)]).to(device)        #(num_chains, Nswap)
    id_uno_swap = torch.stack([torch.randperm(LL - Nocu)[:Nswap] for _ in range(num_chains)]).to(device) + Nocu #(num_chains, LL-Nswap)        
    #id_ocu_swap = torch.multinomial(torch.ones((num_chains, Nocu), device=device), Nswap)            #(num_chains, Nswap)
    #id_uno_swap = torch.multinomial(torch.ones((num_chains, LL - Nocu), device=device), Nswap) + Nocu #(num_chains, LL-Nswap)
    ele_ocu_swap = position[id_ocu_swap, help_id]
    ele_uno_swap = position[id_uno_swap, help_id]
    #ele_ocu_swap = torch.gather(position, 1, id_ocu_swap)
    #ele_uno_swap = torch.gather(position, 1, id_uno_swap)
    #### Swap the elements
    new_position = position.clone()
    new_position[id_ocu_swap, help_id] = ele_uno_swap
    new_position[id_uno_swap, help_id] = ele_ocu_swap
    #new_position.scatter_(1, id_ocu_swap, ele_uno_swap)
    #new_position.scatter_(1, id_uno_swap, ele_ocu_swap)
    return new_position

def help_position2config(positionup: torch.Tensor, positiondown: torch.Tensor, LL: int) -> torch.Tensor:
    """ Convert the position matrix to the config tensor
    
    @param positionup (tensor): The positions of the config of size (Nup, num_chains),
    the first Nup rows record the ids of the occupied sites
    @param positiondown (tensor): The positions of the config of size (Ndown, num_chains),
    @param LL (int): The number of sites
    
    @returns config (tensor): The config tensor of size (num_chains, 2*LL)"""
    num_chains = positionup.size(1)
    config = -torch.ones((2*LL, num_chains), dtype=torch.float64, device=positionup.device)
    config.scatter_(0, positionup, 1.0)
    config.scatter_(0, positiondown, 1.0)
    return config.T

def generate_sample(model: nn.Module, num_samples: int, num_chains: int = 1, drop_rate: float = 0.3, sweep_size: int = None) -> torch.Tensor:
    """ Generate samples from the model.

    @param model (NNB): The neural network model
    @param num_samples (int): The number of samples to generate
    @param num_chains (int): The number of Monte Carlo chains

    @returns samples (Tensor): A tensor of shape (num_samples * num_chains, 2L^2)
    """
    
    # Generate samples
    device = model.device
    LL = model.input_size
    Nup = model.num_fillings[0]
    Ndown = model.num_fillings[1]

    drop_samples = round(num_samples * drop_rate)
    num_samples = num_samples + drop_samples
    samples = torch.zeros(num_samples, num_chains, 2 * LL, dtype=torch.float64, device=device)
    positions = torch.zeros(num_samples, num_chains, Nup+Ndown, dtype=torch.int64, device=device)
    prob_list = torch.zeros(num_samples, num_chains, dtype=torch.complex64, device=device)

    positions_up = torch.stack([torch.randperm(LL) for _ in range(num_chains)], dim=1).to(device)    #(LL, num_chains)
    positions_down = torch.stack([torch.randperm(LL) for _ in range(num_chains)], dim=1).to(device) + LL  #(LL, num_chains)
    prob_old = torch.ones(num_chains)  #(num_chains)

    if sweep_size is None:
        sweep_size = LL * 2

    for i in range(num_samples * sweep_size):
        # Metropolis-Hastings
        Nswap = 1
        new_positions_up = help_flip(positions_up, Nup, Nswap)
        new_positions_down = help_flip(positions_down, Ndown, Nswap)
        # Create the new configuration
        config = help_position2config(new_positions_up[:Nup], new_positions_down[:Ndown], LL) 
        # Create the new configuration
        new_positions = torch.cat((new_positions_up[:Nup].T, new_positions_down[:Ndown].T), dim=1)
             
        # Calculate the acceptance probability
        prob_amp_new = model(config, new_positions)
        prob_new = torch.abs(prob_amp_new) ** 2
        if i == 0:
            prob_old = prob_new
            samples[i,:,:] = config
            prob_list[i] = prob_old
            positions[i,:,:Nup] = new_positions_up[:Nup].T
            positions[i,:,Nup:] = new_positions_down[:Ndown].T
            #positions[i,:,:] = torch.cat((new_positions_up[:Nup].T, new_positions_down[:Ndown].T), dim=1)
        else:
            acceptance = (torch.rand(num_chains, device=device) < prob_new / prob_old).unsqueeze(0)
            positions_up = torch.where(acceptance, new_positions_up, positions_up)
            positions_down = torch.where(acceptance, new_positions_down, positions_down)
            
            prob_old = torch.where(acceptance, prob_new, prob_old).squeeze()
            config = help_position2config(positions_up[:Nup], positions_down[:Ndown], LL)
            
            if i % sweep_size == 0:
                idx = i // sweep_size
                samples[idx,:,:] = config
                prob_list[idx] = prob_old

                positions[idx,:,:Nup] = positions_up[:Nup].T
                positions[idx,:,Nup:] = positions_down[:Ndown].T
                #positions[idx,:,:] = torch.cat((positions_up[:Nup].T, positions_down[:Ndown].T), dim=1)
    samples = samples[drop_samples:].view(-1, 2 * LL)
    positions = positions[drop_samples:].view(-1, Nup + Ndown)
    prob_list = prob_list[drop_samples:].view(-1)
    return samples, positions, prob_list
