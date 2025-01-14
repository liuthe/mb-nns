import torch.profiler

from observable.hamiltonian import Hubbard, batch2state
from observable.correlation import correlation, correlation_nk
from model.nnb_model import*
from sampler.sampler import generate_sample, help_flip, help_position2config

import datetime

import random

random.seed(0)

L = 4
Lx = Ly = L
LL = Lx * Ly
num_fillings = [7, 7]

t = 1
U = 8
H = Hubbard(Lx, Ly, t, U)

n_chains = 16
n_samples = 4096 // n_chains
sweep_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

date = "2024_12_14_01_43"
model = spinsep_NNB.load("./runs/nnb_cuda/" +date+ "/" +date+ "model_simple_cuda.bin")

h_model = Hubbard(Lx, Ly, t=t, U=U)

model.to(device)
lr = 1e-2
train_iter = 5
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_iter, gamma=0.2)

config, position, prob = generate_sample(model, n_samples, n_chains, sweep_size = sweep_size)

loc_e, energy, var_e = h_model.local_energy(config, position, model)
print(f"The energy is {energy}")
print(f"The variance is {var_e}")

with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/spinsep'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:
    for step in range(train_iter):
        prof.step()
        model(config, position)
        #config, position, prob = generate_sample(model, n_samples, n_chains, sweep_size = sweep_size)

# for i in range(train_iter):
#     begin_time = datetime.datetime.now()
#     samples, position, prob = generate_sample(model, n_samples, n_chains, sweep_size = sweep_size)
#     loc_e, energy, var_e = h_model.local_energy(samples, position, model, train = True)
#     # Compute the loss
#     loss = loc_e
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     end_time = datetime.datetime.now()
#     print(f"The step time is {end_time - begin_time}")
