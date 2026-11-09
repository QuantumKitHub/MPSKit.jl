md"""
# The transverse-field Ising model

In this example we study the quantum phase transition of the transverse-field Ising model, by scanning the field across it and comparing finite chains of increasing length with a calculation directly in the thermodynamic limit.
The Hamiltonian is given by

```math
H = -J (∑_{⟨i,j⟩} σᶻᵢ σᶻⱼ + g ∑ᵢ σˣᵢ)
```

where ``J`` sets the energy scale and the field ``g`` tunes the competition between the ``σᶻσᶻ`` interaction and the transverse ``σˣ`` field.
The model has a quantum critical point at ``g = 1``, separating a ferromagnetic phase for ``g < 1`` from a paramagnetic phase for ``g > 1``.
We characterize the transition through the order parameter, the entanglement entropy and the correlation length, and finally extract the central charge of the critical point.
To follow this example you need the following packages:
"""

using MPSKit, TensorKit, CairoMakie
include(joinpath(@__DIR__, "..", "..", "..", "docs", "src", "assets", "models.jl")) #hide
using .ExampleModels
using TensorKitTensors.SpinOperators: σᶻ, S_z_S_z

# For reproducibility of this page, we fix the seed of the random number generator:

using Random
Random.seed!(1234);

md"""
We consider open chains of three different lengths, together with the infinite chain, and scan the field from ``g = 0`` to ``g = 2``.
All states have bond dimension ``D = 32``.
"""

Ls = [32, 64, 128]
D = 32
g_values = 0.0:0.1:2.0

md"""
## Finite chains

The Hamiltonian is invariant under the global spin flip ``∏ᵢ σˣᵢ``, which we impose by working with ``ℤ₂``-symmetric tensors.
On a finite chain, the ground state then is the symmetric superposition of the two ordered states, and ``⟨σᶻᵢ⟩ = 0`` exactly, even in the ordered phase.
Spontaneous symmetry breaking only happens in the thermodynamic limit, so instead we obtain the order parameter from the correlation function between two distant sites, ``m = \sqrt{|⟨σᶻᵢ σᶻⱼ⟩|}``, with ``i = L/4`` and ``j = 3L/4 + 1``.
For every length and field, we find the ground state with DMRG, and keep the states at the critical point for later use.
"""

ZZ = 4 * S_z_S_z(ComplexF64, Z2Irrep; spin = 1 // 2)
P = Z2Space(0 => 1, 1 => 1)
V = Z2Space(0 => D ÷ 2, 1 => D ÷ 2)

m_finite = Dict{Int, Vector{Float64}}()
S_finite = Dict{Int, Vector{Float64}}()
ψ_critical = Dict{Int, FiniteMPS}()
for L in Ls
    m_finite[L] = Float64[]
    S_finite[L] = Float64[]
    for g in g_values
        H = transverse_field_ising(Z2Irrep; L, g)
        ψ, envs, info = find_groundstate(FiniteMPS(L, P, V), H, DMRG(; verbosity = 0))
        push!(m_finite[L], sqrt(abs(correlator(ψ, ZZ, L ÷ 4, 3L ÷ 4 + 1))))
        push!(S_finite[L], real(entropy(ψ, L ÷ 2)))
        g == 1 && (ψ_critical[L] = ψ)
    end
end

md"""
## Infinite chain

In the thermodynamic limit we optimize an `InfiniteMPS` with VUMPS, this time without imposing the symmetry.
In the ordered phase the infinite state then spontaneously breaks it, and we can measure ``|⟨σᶻ⟩|`` directly.
"""

ψs_infinite = map(g_values) do g
    ψ, envs, info = find_groundstate(InfiniteMPS(ℂ^2, ℂ^D), transverse_field_ising(; g), VUMPS(; verbosity = 0))
    return ψ
end
M_infinite = [abs(expectation_value(ψ, 1 => σᶻ())) for ψ in ψs_infinite]
S_infinite = [real(only(entropy(ψ))) for ψ in ψs_infinite];

md"""
## Order parameter

Comparing the finite chains with the infinite chain, the order parameter vanishes at ``g = 1``.
The finite chains approach the infinite result as ``L`` grows, and the transition is rounded off over a window around ``g = 1`` that shrinks with system size.
"""

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "g", ylabel = "order parameter")
for L in Ls
    scatterlines!(ax, g_values, m_finite[L]; label = "L = $L")
end
scatterlines!(ax, g_values, M_infinite; label = "L = ∞", color = :black)
vlines!(ax, [1.0]; color = :gray, linestyle = :dash)
axislegend(ax; position = :rt)
fig

md"""
## Entanglement entropy

At a critical point correlations become long-ranged, and the ground state is more entangled than deep in either phase.
We compare the entanglement entropy across the middle bond of the finite chains with that of the infinite chain.
"""

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "g", ylabel = "S")
for L in Ls
    scatterlines!(ax, g_values, S_finite[L]; label = "L = $L")
end
scatterlines!(ax, g_values, S_infinite; label = "L = ∞", color = :black)
hlines!(ax, [log(2)]; color = :gray, linestyle = :dot)
vlines!(ax, [1.0]; color = :gray, linestyle = :dash)
axislegend(ax; position = :rt)
fig

md"""
In the paramagnetic phase all curves coincide, since the correlation length is much shorter than the chains.
In the ordered phase, the symmetric finite-chain ground state carries an additional entropy ``\log 2`` (dotted line), from the superposition of the two ordered states, which is absent in the symmetry-broken infinite state.
At ``g = 1`` the entropy keeps increasing with the system size, since at the critical point it diverges logarithmically with ``L``, which we will use below to extract the central charge.
"""

md"""
## Correlation length

The correlation length is extracted from the transfer matrix spectrum of the infinite state.
Away from the critical point the ground state has much less entanglement than a bond dimension of ``D = 32`` can capture, and the smallest Schmidt values of the state are numerical noise.
Since these would contaminate the transfer matrix spectrum, we first truncate them with `changebonds` and `SvdCut`.
At ``g = 0`` the ground state is an exact product state, for which the correlation length is not defined, so we leave out this point.
"""

ξ_infinite = map(ψs_infinite[2:end]) do ψ
    ψ_trunc = changebonds(ψ, SvdCut(; trunc = trunctol(; atol = 1.0e-10)))
    return only(values(correlation_length(ψ_trunc)))
end

fig, ax, _ = scatterlines(g_values[2:end], ξ_infinite; axis = (; xlabel = "g", ylabel = "ξ", yscale = log10))
vlines!(ax, [1.0]; color = :gray, linestyle = :dash)
fig

md"""
The correlation length grows towards ``g = 1`` from both sides.
At the critical point it should diverge, but an MPS with finite bond dimension can only capture correlations up to a finite length, so the peak grows as ``D`` is increased.
"""

md"""
## Central charge

At the critical point, the transverse-field Ising model is described by a conformal field theory with central charge ``c = 1/2``.
For an open chain of length ``L``, the entanglement entropy across bond ``l`` then follows [calabrese2004](@cite)

```math
S(l) = \frac{c}{6} \log\left[\frac{2L}{π} \sin\left(\frac{π l}{L}\right)\right] + \text{const.}
```

Plotting the entropy of every bond against the logarithm on the right-hand side, the data for all system sizes should fall onto a single line with slope ``c/6``.
"""

x_values = Float64[]
S_values = Float64[]
fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "log[(2L/π) sin(πl/L)]", ylabel = "S(l)")
for L in Ls
    x = [log(2L / π * sin(π * l / L)) for l in 1:(L - 1)]
    S = [real(entropy(ψ_critical[L], l)) for l in 1:(L - 1)]
    scatter!(ax, x, S; label = "L = $L")
    append!(x_values, x)
    append!(S_values, S)
end
axislegend(ax; position = :lt)
fig

md"""
A linear fit of all data points gives an estimate of the central charge:
"""

slope, offset = [x_values ones(length(x_values))] \ S_values
c = 6 * slope

md"""
This is close to the exact value ``c = 1/2``.
The remaining deviation stems mostly from the bonds near the edges of the chain, where the logarithmic form only holds asymptotically.
The critical point is studied in more detail in the [Ising CFT](@ref "The Ising CFT spectrum") example, which matches its excitation spectrum to the predictions of conformal field theory.
"""
