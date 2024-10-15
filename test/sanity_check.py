# This file is used to check the sanity of my model
import sys
sys.path.append("..")
from model.nnb_model import NNB
import torch

sys_size = 16
hidden_size = 10
num_fillings = [8,8]
num_orbitals = 2

test_model = NNB(sys_size, hidden_size, num_fillings)

print("construct successfully!")

config = torch.ones(10, 2 * sys_size, dtype=torch.float)
out_up, out_down = test_model.forward(config)
print("Forward pass successfully!")

sample, prob_list = test_model.generate_sample(10)
print(torch.sum(sample, 1))
print(prob_list)
print("Sample generation successfully!")
