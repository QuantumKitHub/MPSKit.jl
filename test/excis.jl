using MPSKit, TensorKit
import LinearAlgebra.eigvals
using MPSKit: max_virtualspaces

V = U1Space(i => 1 for i in 0:1)
O = randn(V^2 ← V^2)
O += O'  # Hermitian

N = 3
H = FiniteMPOHamiltonian(fill(V, N), (i, i + 1) => O for i in 1:(N - 1));

h = convert(TensorMap, H);

sectors = collect(blocksectors(h))
sec = U1Irrep(1)
@assert sec in sectors
num = 10
vals1 = eigvals(block(h, sec))
vals2 = eigvals(h)[sec]
@assert vals1 ≈ vals2
vals3, vecs = exact_diagonalization(H; sector=(sec), num);
vals1
vals3

right = U1Space(sec => 1)
max_virtualspaces(physicalspace(H); right)

psi_full = rand(oneunit(V) * V^N ← right);
dim(space(psi_full))
psi = MPSKit.decompose_localmps(psi_full);
left_virtualspace.(psi)

left_virtualspace.(vecs[1].AL)
right_virtualspace.(vecs[1].AL)
max_virtualspaces(physicalspace(H); right)
psi1 = FiniteMPS(physicalspace(H), max_virtualspaces(physicalspace(H))[2:(end - 1)])
psi1, = find_groundstate(psi1, H);
psi2 = FiniteMPS(physicalspace(H), max_virtualspaces(physicalspace(H); right)[2:(end - 1)];
                 right)
left_virtualspace.(psi2.AL)
right_virtualspace.(psi2.AL)

psi2, = find_groundstate(psi2, H);
expectation_value(psi2, H)

Es, Bs = excitations(H, QuasiparticleAnsatz(), FiniteMPS(psi); sector=sec);
@inferred excitations(H, QuasiparticleAnsatz(), FiniteMPS(psi); sector=sec);

Es .+ expectation_value(psi1, H)

vals1
vals3

psi2
using TestEnv
using Test
TestEnv.activate()
include("setup.jl")
using .TestSetup
using TensorKit, MPSKit
using MPSKit: Multiline
using KrylovKit

H = repeat(TestSetup.sixvertex(), 2)
ψ = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^10, ℂ^10])
ψ, envs, _ = leading_boundary(ψ, H,
                              VUMPS(; maxiter=400, tol=1e-10))
energies, ϕs = @inferred excitations(H, QuasiparticleAnsatz(),
                                     [0.0, Float64(pi / 2)], ψ,
                                     envs; verbosity=0)
@test abs(energies[1]) > abs(energies[2]) # has a minimum at pi/2
alg = QuasiparticleAnsatz()
ps = [0.0, Float64(pi / 2)]
excitations(H, alg, ps, ψ, envs; verbosity=0);
using Cthulhu
Hm = convert(MultilineMPO, H);
psim = convert(MultilineMPS, ψ);
envs = environments(psim, Hm);
excitations(Hm, alg, ps[1], psim, envs, psim, envs; verbosity=0);

@descend excitations(Hm, alg, ps[1], psim, envs, psim, envs);

qp = Multiline([LeftGaugedQP(rand, psim[1], psim[1]; sector=one(sectortype(psim[1])),
                             momentum=ps[1])]);
excitations(Hm, alg, qp, envs, envs; num=1);

@descend excitations(Hm, alg, qp, envs, envs; num=1);
@code_warntype excitations(Hm, alg, qp, envs, envs; num=1);

Heff = MPSKit.EffectiveExcitationHamiltonian(H, envs[1], envs[1], fill(1.0, length(H)));
Heffs = Multiline([Heff]);
@code_warntype KrylovKit.apply(Heff, qp[1])
@descend KrylovKit.apply(Heff, qp[1])

KrylovKit.apply(Multiline([Heff]), qp);
@code_warntype KrylovKit.apply(Heffs, qp);

@descend eigsolve(Heffs, qp, 1, :LM, Lanczos());
@code_warntype eigsolve(Heffs, qp, 1, :LM);
@descend eigsolve(Heffs, qp, 1, :LM);
