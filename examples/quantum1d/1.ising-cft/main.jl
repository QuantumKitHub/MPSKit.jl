md"""
# The Ising CFT spectrum

This tutorial is meant to show the finite size CFT spectrum for the quantum Ising model. We
do this by first employing an exact diagonalization technique, and then extending the
analysis to larger system sizes through the use of MPS techniques.
"""

using MPSKit, MPSKitModels, TensorKit, Plots, KrylovKit
using LinearAlgebra: eigen, diagm, Hermitian

md"""
The hamiltonian is defined on a finite lattice with periodic boundary conditions,
which can be implemented as follows:
"""

L = 12
H = periodic_boundary_conditions(transverse_field_ising(), L);

md"""
## Exact diagonalisation

In MPSKit, there is support for exact diagonalisation by leveraging the fact that applying
the hamiltonian to an untruncated MPS will result in an effective hamiltonian on the center
site which implements the action of the entire hamiltonian. Thus, optimizing the middle
tensor is equivalent to optimixing a state in the entire Hilbert space, as all other tensors
are just unitary matrices that mix the basis.
"""

energies, states = exact_diagonalization(H; num=18, alg=Lanczos(; krylovdim=200));
plot(real.(energies);
     seriestype=:scatter,
     legend=false,
     ylabel="energy",
     xlabel="#eigenvalue")

md"""
!!! note "Krylov dimension"
    Note that we have specified a large Krylov dimension as degenerate eigenvalues are
    notoriously difficult for iterative methods.
"""

md"""
## Extracting momentum

Given a state, it is possible to assign a momentum label
through the use of the translation operator. This operator can be defined in MPO language
either diagramatically as

![translation operator MPO](translation_mpo.png)

or in the code as:
"""

id = complex(isomorphism(ℂ^2, ℂ^2))
@tensor O[-1 -2; -3 -4] := id[-1, -3] * id[-2, -4]
T = periodic_boundary_conditions(DenseMPO(O), L);

md"""
We can then calculate the momentum of the groundstate as the expectation value of this
operator. However, there is a subtlety because of the degeneracies in the energy
eigenvalues. The eigensolver will find an orthonormal basis within each energy subspace, but
this basis is not necessarily a basis of eigenstates of the translation operator. In order
to fix this, we diagonalize the translation operator within each energy subspace.
"""

momentum(ψᵢ, ψⱼ=ψᵢ) = angle(dot(ψᵢ, T * ψⱼ))

function fix_degeneracies(basis)
    N = zeros(ComplexF64, length(basis), length(basis))
    M = zeros(ComplexF64, length(basis), length(basis))
    for i in eachindex(basis), j in eachindex(basis)
        N[i, j] = dot(basis[i], basis[j])
        M[i, j] = momentum(basis[i], basis[j])
    end

    vals, vecs = eigen(Hermitian(N))
    M = (vecs' * M * vecs)
    M /= diagm(vals)

    vals, vecs = eigen(M)
    return angle.(vals)
end

momenta = Float64[]
append!(momenta, fix_degeneracies(states[1:1]))
append!(momenta, fix_degeneracies(states[2:2]))
append!(momenta, fix_degeneracies(states[3:3]))
append!(momenta, fix_degeneracies(states[4:5]))
append!(momenta, fix_degeneracies(states[6:9]))
append!(momenta, fix_degeneracies(states[10:11]))
append!(momenta, fix_degeneracies(states[12:12]))
append!(momenta, fix_degeneracies(states[13:16]))
append!(momenta, fix_degeneracies(states[17:18]))

plot(momenta,
     real.(energies[1:18]);
     seriestype=:scatter,
     xlabel="momentum",
     ylabel="energy",
     legend=false)

md"""
## Finite bond dimension

If we limit the maximum bond dimension of the MPS, we get an approximate solution, but we
can reach higher system sizes.
"""

L_mps = 20
H_mps = periodic_boundary_conditions(transverse_field_ising(), L_mps)
D = 64
ψ, envs, δ = find_groundstate(FiniteMPS(L_mps, ℂ^2, ℂ^D), H_mps, DMRG());

md"""
Excitations on top of the groundstate can be found through the use of the quasiparticle
ansatz. This returns quasiparticle states, which can be converted to regular `FiniteMPS`
objects.
"""

E_ex, qps = excitations(H, QuasiparticleAnsatz(), ψ, envs; num=16)
states_mps = vcat(ψ, map(qp -> convert(FiniteMPS, qp), qps))
E_mps = map(x -> sum(expectation_value(x, H_mps)), states_mps)

T_mps = periodic_boundary_conditions(DenseMPO(O), L_mps)
momenta_mps = Float64[]
append!(momenta_mps, fix_degeneracies(states[1:1]))
append!(momenta_mps, fix_degeneracies(states[2:2]))
append!(momenta_mps, fix_degeneracies(states[3:3]))
append!(momenta_mps, fix_degeneracies(states[4:5]))
append!(momenta_mps, fix_degeneracies(states[6:9]))
append!(momenta_mps, fix_degeneracies(states[10:11]))
append!(momenta_mps, fix_degeneracies(states[12:12]))
append!(momenta_mps, fix_degeneracies(states[13:16]))

plot(momenta_mps,
     real.(energies[1:16]);
     seriestype=:scatter,
     xlabel="momentum",
     ylabel="energy",
     legend=false)
