import sys
sys.path.append("..")
import torch
from model.nnb_model import NNB
from observable.hamiltonian import batch2state, Hubbard

# Set the basic parameter
L = 4
t = 1
U = 1

# Construct the NNB state
sys_size = L * L
hidden_size = 100
num_fillings = [8,8]

test_model = NNB(sys_size, hidden_size, num_fillings)

# Construct the Hamiltonian
hubbard_model = Hubbard(L, t, U)

# Generate a Metropolis sample
sample, prob_list = test_model.generate_sample(1000)
#print("The sample is: ", sample)
# Calculate the energy
energy = hubbard_model.energy_local(sample, test_model)
print("The energy is: ", energy)

