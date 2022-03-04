using MPSKit,MPSKitModels,TensorKit

mpo = nonsym_ising_mpo();

state = InfiniteMPS([ℂ^2],[ℂ^10]);

(state,envs,_) = leading_boundary(state,mpo,VUMPS(tol_galerkin=1e-10));
