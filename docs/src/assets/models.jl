# Model constructors for the MPSKit documentation and examples.
# Adapted from MPSKitModels.jl v0.4.7, src/models/{hamiltonians,transfermatrices}.jl:
# https://github.com/QuantumKitHub/MPSKitModels.jl/tree/v0.4.7/src/models
# Operators come from TensorKitTensors; chain assembly uses MPSKit directly.
#
# MIT License
# Copyright (c) 2021 Maarten Van Damme
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

module ExampleModels

using MPSKit, TensorKit
using TensorKitTensors: SpinOperators, BosonOperators, HubbardOperators

export chain_hamiltonian, transverse_field_ising, heisenberg_XXX
export hubbard_model, bose_hubbard_model, hard_hexagon

# L specifies an open finite chain; otherwise unitcell specifies an infinite chain.
function chain_hamiltonian(twosite, onesite = nothing; L = nothing, unitcell::Integer = 1)
    nsites = isnothing(L) ? unitcell : L
    nsites isa Integer && nsites > 0 || throw(ArgumentError("chain length must be a positive integer"))
    nbonds = isnothing(L) ? nsites : nsites - 1
    terms = Pair[(i, i + 1) => twosite for i in 1:nbonds]
    if !isnothing(onesite)
        append!(terms, [i => onesite for i in 1:nsites])
    end
    spaces = fill(space(twosite, 1), nsites)
    constructor = isnothing(L) ? InfiniteMPOHamiltonian : FiniteMPOHamiltonian
    return constructor(spaces, terms)
end

# H = -J sum(σz_i σz_{i+1} + g σx_i).
function transverse_field_ising(
        T::Type{<:Number} = ComplexF64, symmetry::Type{<:Sector} = Trivial;
        J = 1.0, g = 1.0, kwargs...
    )
    ZZ = 4 * SpinOperators.S_z_S_z(T, symmetry; spin = 1 // 2)
    X = SpinOperators.σˣ(T, symmetry)
    return chain_hamiltonian(-J * ZZ, -J * g * X; kwargs...)
end
transverse_field_ising(symmetry::Type{<:Sector}; kwargs...) =
    transverse_field_ising(ComplexF64, symmetry; kwargs...)

# H = J sum(S_i ⋅ S_{i+1}); preserve the examples' spin-1 default.
function heisenberg_XXX(
        T::Type{<:Number} = ComplexF64, symmetry::Type{<:Sector} = Trivial;
        J = 1.0, spin = 1, kwargs...
    )
    term = J * SpinOperators.S_exchange(T, symmetry; spin)
    return chain_hamiltonian(term; kwargs...)
end
heisenberg_XXX(symmetry::Type{<:Sector}; kwargs...) =
    heisenberg_XXX(ComplexF64, symmetry; kwargs...)

# H = -t sum(e⁺_i e⁻_{i+1} + h.c.) + U sum(n↑ n↓) - mu sum(n).
function hubbard_model(
        T::Type{<:Number} = ComplexF64, particle_symmetry::Type{<:Sector} = Trivial,
        spin_symmetry::Type{<:Sector} = Trivial;
        t = 1.0, U = 1.0, mu = 0.0, kwargs...
    )
    # e_hopping includes the fermionic reordering sign and is Hermitian.
    hopping = HubbardOperators.e_hopping(T, particle_symmetry, spin_symmetry)
    interaction = HubbardOperators.ud_num(T, particle_symmetry, spin_symmetry)
    N = HubbardOperators.e_num(T, particle_symmetry, spin_symmetry)
    return chain_hamiltonian(-t * hopping, U * interaction - mu * N; kwargs...)
end

# H = -t sum(b⁺_i b⁻_{i+1} + h.c.) + U/2 sum(n(n-1)) - mu sum(n).
function bose_hubbard_model(
        T::Type{<:Number} = ComplexF64, symmetry::Type{<:Sector} = Trivial;
        cutoff::Integer = 5, t = 1.0, U = 1.0, mu = 0.0, kwargs...
    )
    hopping = BosonOperators.b_plus_b_min(T, symmetry; cutoff) +
        BosonOperators.b_min_b_plus(T, symmetry; cutoff)
    N = BosonOperators.b_num(T, symmetry; cutoff)
    interaction = N * (N - id(domain(N)))
    return chain_hamiltonian(-t * hopping, U / 2 * interaction - mu * N; kwargs...)
end

function hard_hexagon(T::Type{<:Number} = ComplexF64)
    P = Vect[FibonacciAnyon](:τ => 1)
    O = ones(T, P ⊗ P ← P ⊗ P)
    block(O, FibonacciAnyon(:I)) .*= 0
    return InfiniteMPO([O])
end

end
