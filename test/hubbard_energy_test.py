# This file is used to check the sanity of my model
import sys
sys.path.append("..")
#sys.path.append("../model")
from model.nnb_model import NNB
from observable.hamiltonian import Hubbard, config2state
import torch

L = 4
t = 1
U = 1

hubbard_model = Hubbard(L, t, U)
print("construct successfully!")
# The first state test
r = torch.ones(2 * L * L, dtype=torch.bool)
psi = config2state(r)

e1 = hubbard_model.energy_local_single(r, psi)
print("The local energy is: ", e1)
print("The local energy should be: ", U*L*L)

# The second state test
r = torch.ones(2 * L * L, dtype=torch.bool)
r[0] = 0
psi = config2state(r)
e2 = hubbard_model.energy_local_single(r, psi)
print("The local energy is: ", e2)
print("The local energy should be: ", U*(L*L-1))

