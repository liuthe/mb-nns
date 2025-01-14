import math
import sys
import pickle
import datetime
from pathlib import Path

from model.nnb_model import*
from observable.hamiltonian import Hubbard
from sampler.sampler import generate_sample

from utils.batch_utils import batch_iter

import torch
import torch.nn.utils
from torch.utils.tensorboard import SummaryWriter

# Set the basic parameter
epochs = 7
train_iter = 600
train_batch_size = 4096
uniform_init = 0.5
lr = 1e-2
model_save_path = "model_simple_cuda.bin"

n_chains = 16
n_samples = train_batch_size // n_chains

t = 1
U = 8
L = 4
Lx = Ly = L
sys_size = Lx * Ly
hidden_size = 128
num_fillings = [7, 7]
sweep_size = sys_size * 2

# Construct the NNB state
h_model = Hubbard(Lx, Ly, t=t, U=U)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

date = "2024_12_14_01_43"
model = spinsep_NNB.load("./runs/nnb_cuda/" +date+ "/" +date+ "model_simple_cuda.bin")
model.to(device)
# Set the tensorboard writer
current_time = datetime.datetime.now()
time_str = current_time.strftime("%Y_%m_%d_%H_%M")
tensorboard_path = "nnb_cuda" 
writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}/{time_str}")
# Begin the train mode
model.train()

#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Switch to SGD
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_iter, gamma=0.2)


for epoch in range(epochs):
    for i in range(train_iter):
        begin_time = datetime.datetime.now()
        samples,positions, prob = generate_sample(model, n_samples, n_chains, sweep_size = sweep_size)
        loc_e, energy, var_e = h_model.local_energy(samples, positions, model)
        # Compute the loss
        loss = loc_e
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('Loss/train', loss, epoch * train_iter + i)
        writer.add_scalar('Energy/local', energy, epoch * train_iter + i)
        writer.add_scalar('Energy/variance', var_e, epoch * train_iter + i)

        end_time = datetime.datetime.now()
        print(f"Epoch {epoch+1}, Iter {i+1}, Loss {loss}, Energy {energy}, Time {end_time - begin_time}")
    scheduler.step()
    torch.cuda.empty_cache()
model.save(f"./runs/{tensorboard_path}/{time_str}/"+time_str + model_save_path)
torch.save(optimizer.state_dict(), f"./runs/{tensorboard_path}/{time_str}/"+time_str + model_save_path + '.optim')
