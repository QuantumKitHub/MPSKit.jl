using MPSKit, MPSKitModels, TensorKit

L = 30
Dfinal = 50
H = periodic_boundary_conditions(nonsym_xxz_ham(), L);
state = FiniteMPS(L, ℂ^3, ℂ^10);

println("Controlled Bond Expansion:")
find_groundstate(state, H, CBE_DMRG(; Dfinal=Dfinal, maxiter=10));
println("DMRG2:")
find_groundstate(state, H, DMRG2(; trscheme=truncdim(Dfinal), maxiter=10));
