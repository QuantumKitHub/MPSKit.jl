using MPSKit, MPSKitModels, TensorKit

L = 20
H = periodic_boundary_conditions(nonsym_ising_ham(), L);
state = FiniteMPS(L, ℂ^2, ℂ^4);

alg = CBE_DMRG()
state, envs, delta = find_groundstate(state, H, alg);
