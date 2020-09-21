using MPSKit,MPSKitModels,TensorKit

mpo = nonsym_ising_mpo();
state = InfiniteMPS([ℂ^2],[ℂ^10]);
(state,pars,_) = leading_boundary(state,mpo,Vumps(tol_galerkin=1e-10));

(state,pars,_) = leading_boundary(state,mpo,PowerMethod(tol_galerkin=1e-12, maxiter=400));
