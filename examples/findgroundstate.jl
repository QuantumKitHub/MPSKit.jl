using MPSKit,MPSKitModels,TensorKit,Test

#defining the hamiltonian
th = nonsym_ising_ham(lambda = 4.0);
szt = TensorMap([1 0;0 -1],ℂ^2,ℂ^2)

# ------------------------------------
# |     Drmg                         |
# ------------------------------------
ts = FiniteMPS(10,ℂ^2,ℂ^10);
(ts,envs,_) = find_groundstate(ts,th,DMRG());

szval_finite = sum(expectation_value(ts,szt))/length(ts)
@test szval_finite ≈ 0 atol=1e-12

# ------------------------------------
# |     Drmg2                        |
# ------------------------------------
ts = FiniteMPS(fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10));
(ts,envs,_) = find_groundstate(ts,th,DMRG2(trscheme = truncdim(15)));

szval_finite = sum(expectation_value(ts,szt))/length(ts)
@test szval_finite ≈ 0 atol=1e-12

# ------------------------------------
# |     Vumps                        |
# ------------------------------------
ts = InfiniteMPS([ℂ^2],[ℂ^50]);
(ts,envs,_) = find_groundstate(ts,th,VUMPS(maxiter=400));

szval_infinite = sum(expectation_value(ts,szt))/length(ts)
@test szval_infinite ≈ 0 atol=1e-12

# ------------------------------------
# |     Gradient optimization        |
# ------------------------------------
ts = InfiniteMPS([ℂ^2], [ℂ^5]);
(ts, envs, _) = find_groundstate(ts, th, GradientGrassmann(maxiter=400));

szval_infinite = sum(expectation_value(ts,szt))/length(ts)
@test szval_infinite ≈ 0 atol=1e-12
