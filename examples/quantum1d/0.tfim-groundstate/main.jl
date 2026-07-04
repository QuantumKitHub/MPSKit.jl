md"""
# The transverse-field Ising model: a complete ground-state study

This example is the bridge from the introductory tutorials into the research-grade
gallery.
If you have worked through [Your first ground state](@ref tutorial_first_groundstate) and
[The thermodynamic limit](@ref tutorial_thermodynamic_limit) you already know every
individual tool used here; the goal now is to *assemble* them into one coherent case
study of a genuine quantum phase transition.

We use the same transverse-field Ising model (TFIM) as the tutorials, on a chain of
spin-1/2 sites:

```math
H = -J\left(\sum_{\langle i,j\rangle} \sigma^z_i \sigma^z_j + g\sum_i \sigma^x_i\right),
```

where the first sum runs over neighbouring pairs, ``J`` sets the energy scale, and the
dimensionless field ``g`` tunes the competition between the ``\sigma^z\sigma^z``
interaction and the transverse ``\sigma^x`` field.
The model has a quantum critical point at ``g = 1``.

<!-- REVIEW: physics framing — critical point at g = 1 for this J(ZZ) + Jg(X) convention, and the "competition between interaction and field" description. Please confirm (mirrors the framing already used in the tutorials). -->

Rather than looking at a single field value, we will scan ``g`` across the transition and
diagnose it three independent ways, comparing a *finite* chain against a calculation
performed *directly in the thermodynamic limit*:

1. the order parameter ``|\langle\sigma^z\rangle|``, computed both for a finite ring and
   for an infinite chain, in one figure;
2. the entanglement entropy of the infinite state;
3. the correlation length of the infinite state.

All three should point at the same place — that agreement is the payoff.
"""

# We take the model and lattice from MPSKitModels, the tensor backend from TensorKit, and
# Plots for the figures. The Pauli operators `σᶻ`, `σˣ` are re-exported by MPSKitModels.

using MPSKit, MPSKitModels, TensorKit, Plots

md"""
## Shared parameters

We fix a finite chain length `L`, a bond dimension `D` (the accuracy knob, see
[Controlling bond dimension](@ref howto_bond_dimension)), and the set of field values to
scan.
`D` is kept modest so the whole page runs in a couple of minutes; increasing it sharpens
every curve below.

<!-- REVIEW: expected-behavior claim — increasing the bond dimension D sharpens the transition in all three diagnostics (magnetization, entropy, correlation length). Please confirm this qualitative statement. -->
"""

L = 16
D = 8
g_values = 0.1:0.1:2.0

md"""
## 1. Finite versus infinite magnetization

We compute the order parameter ``|\langle\sigma^z\rangle|`` two ways at every field value.

For the **finite** calculation we place the chain on a ring with
[`periodic_boundary_conditions`](@ref) — this removes the open-end boundary effects and
gives a cleaner curve at fixed `L` — and optimize with [`DMRG`](@ref).
We average ``\langle\sigma^z_i\rangle`` over the sites and take the absolute value, because
a finite chain does not spontaneously break its symmetry and the raw sum can land on either
sign (see the discussion in [Your first ground state](@ref tutorial_first_groundstate)).
"""

ψ₀_finite = FiniteMPS(L, ℂ^2, ℂ^D)
M_finite = map(g_values) do g
    H = periodic_boundary_conditions(transverse_field_ising(; g = g), L)
    ψ, = find_groundstate(ψ₀_finite, H, DMRG(; verbosity = 0))
    return abs(sum(expectation_value(ψ, i => σᶻ()) for i in 1:L)) / L
end;

md"""
For the **infinite** calculation we drop the lattice argument to build the Hamiltonian on
the infinite chain, use an [`InfiniteMPS`](@ref), and optimize with [`VUMPS`](@ref).
We keep every optimized infinite state, because we will reuse them for the entropy and
correlation-length diagnostics below.
"""

ψ₀_infinite = InfiniteMPS(ℂ^2, ℂ^D)
states_infinite = map(g_values) do g
    H = transverse_field_ising(; g = g)
    ψ, = find_groundstate(ψ₀_infinite, H, VUMPS(; verbosity = 0))
    return ψ
end;

md"""
The order parameter of a translation-invariant state is just ``\langle\sigma^z\rangle`` on
a single site of the unit cell; we again take the absolute value.

<!-- REVIEW: for the infinite state on the ordered side (g < 1) the finite-D MPS can settle into one of the two symmetry-broken ground states, so abs collapses the two branches onto one curve. This is the same subtlety flagged in the thermodynamic-limit tutorial; please confirm. -->
"""

M_infinite = [abs(expectation_value(ψ, 1 => σᶻ())) for ψ in states_infinite];

md"""
Plotting both curves in a single figure lets us compare them directly.
"""

p_magnetization = plot(;
    xlabel = "g", ylabel = "|⟨σᶻ⟩|", title = "TFIM order parameter", legend = :bottomleft
)
scatter!(p_magnetization, g_values, M_finite; label = "finite ring, L = $L, D = $D")
scatter!(p_magnetization, g_values, M_infinite; label = "infinite, D = $D")
vline!(p_magnetization, [1.0]; color = "gray", linestyle = :dash, label = "g = 1")
p_magnetization

md"""
Both curves are large on the ordered side (small `g`) and fall toward zero on the
disordered side (large `g`), with the crossover near `g = 1`.
The finite ring rounds the transition off into a smooth crossover whose apparent location
is shifted away from `g = 1`, while the infinite calculation drops much more steeply near
the critical point because there is no finite size to smear it out.

<!-- REVIEW: expected-behavior claim — the finite (L = 16) curve is a rounded crossover shifted from g = 1, while the infinite (VUMPS, finite D) curve is sharper and sits closer to g = 1. Neither is a true sharp transition at these finite L / finite D. Please confirm. -->
"""

md"""
## 2. Entanglement entropy across the transition

Entanglement is a hallmark of criticality: it is bounded away from the critical point but
grows sharply as we approach it.
For an [`InfiniteMPS`](@ref), [`entropy`](@ref) returns the von Neumann entanglement entropy
per bond, one value for each site of the unit cell.
Our unit cell has a single site, so we take the one entry with `only`.

<!-- REVIEW: assumes entropy(::InfiniteMPS) returns a vector with one entry per unit-cell site, hence length 1 for a single-site cell; only(...) then extracts the scalar. Please confirm the return shape. -->
"""

S_infinite = [real(only(entropy(ψ))) for ψ in states_infinite]
p_entropy = scatter(
    g_values, S_infinite;
    xlabel = "g", ylabel = "entanglement entropy S", title = "TFIM entanglement entropy",
    legend = false
)
vline!(p_entropy, [1.0]; color = "gray", linestyle = :dash)
p_entropy

md"""
The entropy peaks near `g = 1`.
That peak is the entanglement signature of the phase transition: at criticality
correlations become long-ranged and the ground state is maximally entangled, whereas deep
in either phase the state is closer to a simple product and the entropy is small.

<!-- REVIEW: physics claim — the entanglement entropy peaks at/near the critical point g = 1, and this peak is the entanglement signature of the transition. Please confirm the location and interpretation. -->
"""

md"""
## 3. Correlation length across the transition

The [`correlation_length`](@ref) measures how far apart two spins can still influence each
other; it is extracted from the transfer-matrix spectrum of the uniform infinite state and
has no finite-chain analogue.
It grows toward criticality, so we plot it on a logarithmic vertical axis to make the
growth visible.
"""

ξ_infinite = [correlation_length(ψ) for ψ in states_infinite]
p_xi = scatter(
    g_values, ξ_infinite;
    xlabel = "g", ylabel = "correlation length ξ", yscale = :log10,
    title = "TFIM correlation length", legend = false
)
vline!(p_xi, [1.0]; color = "gray", linestyle = :dash)
p_xi

md"""
The correlation length peaks near `g = 1` as well.
At a genuine critical point it would diverge, but a finite bond dimension `D` can only
capture correlations out to a finite range, so what we measure is large-but-capped rather
than infinite — the peak grows and sharpens as `D` is increased.

<!-- REVIEW: physics claim — the correlation length peaks near g = 1 and would diverge at the true critical point, but finite bond dimension D caps it (finite-entanglement scaling), so the measured value near g = 1 is finite. Please confirm. -->
"""

md"""
## What you now have

Three independent diagnostics — the order parameter, the entanglement entropy, and the
correlation length — all locate the transition of the transverse-field Ising model near
`g = 1`, and the finite-versus-infinite comparison shows concretely how working directly in
the thermodynamic limit removes the finite-size rounding.

<!-- REVIEW: summary claim — all three diagnostics agree on a transition near g = 1. Please confirm this is a fair takeaway at these finite L / finite D. -->

From here the gallery goes further.
The Ising CFT example extracts the momentum-resolved excitation spectrum right at
criticality and matches it to the predictions of conformal field theory, turning the "there
is a critical point near `g = 1`" of this page into a quantitative fingerprint of *which*
critical theory it is.
Every curve on this page also sharpens if you rerun it at a larger bond dimension `D`.
"""
