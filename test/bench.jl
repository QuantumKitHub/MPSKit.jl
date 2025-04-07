using TestEnv
TestEnv.activate()

using Profile

include("setup.jl")
using .TestSetup
using TensorKit
using BlockTensorKit
using MPSKit
using MPSKit: ∂∂AC2
using BenchmarkTools
using KrylovKit

L = 50
g = 1
D = 128
H = heisenberg_XXX(; L)
psi = FiniteMPS(randn, ComplexF64, physicalspace(H), fill(ComplexSpace(D), L - 1))
envs = environments(psi, H);

eigalg = Lanczos(; tol=1e-14, krylovdim=5, maxiter=1, verbosity=0, eager=false)
@benchmark find_groundstate(psi, H,
                            DMRG2(; eigalg, trscheme=truncdim(D), maxiter=3, verbosity=0);
                            svd_alg=TensorKit.SVD())
@benchmark find_groundstate(psi, H,
                            DMRG2(; eigalg, trscheme=truncdim(D), maxiter=3, verbosity=0);
                            svd_alg=TensorKit.SDD())
@profview find_groundstate(psi, H,
                           DMRG2(; eigalg, trscheme=truncdim(D), maxiter=3, verbosity=0));

pos = L ÷ 2
Heff = ∂∂AC2(pos, psi, H, envs);
@plansor x0[-1 -2; -3 -4] := psi.AC[pos][-1 -2; 1] * psi.AR[pos + 1][1 -4; -3];

@benchmark $Heff * $x0

H2 = FiniteMPO(map(SparseBlockTensorMap, parent(H)))
envs2 = environments(psi, H2);
H2eff = ∂∂AC2(pos, psi, H2, envs2);
@plansor x0[-1 -2; -3 -4] := psi.AC[pos][-1 -2; 1] * psi.AR[pos + 1][1 -4; -3];

@benchmark $H2eff * $x0

VSCodeServer.view_profile()
