"""
$(TYPEDEF)

Finite MPS time-evolution algorithm based on the Time-Evolving Block Decimation (TEBD) method:
the Hamiltonian is decomposed into local [`ClusterTerm`](@ref)s (via [`cluster_hamiltonians`](@ref)),
each exponentiated into a gate, and the gates are applied to the state through a Suzuki-Trotter
splitting.

Every gate is applied by contracting it onto its site range and immediately splitting the result
back into individual site tensors via `alg_gauge`, truncating the bond(s) touched by that gate
right away. Since the terms within one Trotter layer act on disjoint sites by construction, this
is equivalent to truncating once after applying each whole layer.

## Fields

$(TYPEDFIELDS)
"""
struct TEBD{G, F} <: Algorithm
    "order of the Suzuki-Trotter splitting: `1` (Lie-Trotter) or `2` (symmetric/Strang)"
    order::Int

    "factorization used to split a gate's evolved tensor back into site tensors: a QR algorithm (no truncation) or a truncated SVD"
    alg_gauge::G

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function TEBD(;
        order::Int = 2, alg_orth = Defaults.alg_orth(), finalize = Defaults._finalize,
        trscheme = notrunc(), alg_svd = Defaults.alg_svd()
    )
    order in (1, 2) || throw(ArgumentError("TEBD only supports order = 1 or 2, got $order"))
    # a no-truncation `trscheme` selects a (bond-preserving) QR gauge, anything else a truncated SVD
    alg_gauge = trscheme isa MatrixAlgebraKit.NoTruncation ? alg_orth :
        MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trscheme)
    return TEBD(order, alg_gauge, finalize)
end

# Greedy interval-graph coloring: sort terms by their starting site, then assign each to the
# first layer whose current rightmost occupied site lies before this term's start. Since
# `ClusterTerm.sites` ranges are intervals on a line, this is optimal (the number of layers
# produced equals the maximum number of terms that simultaneously overlap any single site).
#
# `layer_ends[k]` always means "the rightmost site currently occupied in layer `k`" — it's the
# one piece of state that lets the next term decide whether it's safe to reuse that layer. Once a
# layer's occupant ends before the new term starts, that layer is free again regardless of *when*
# it was opened or which term is currently sitting in it, so `findfirst` may reuse any earlier
# layer, not just the most recently touched one; only the running "rightmost site so far" matters
# for correctness (no overlap), not the history of who has passed through a given slot.
function _partition_layers(clusters::Vector{<:ClusterTerm})
    layers = Vector{eltype(clusters)}[]
    layer_ends = Int[]
    for c in sort(clusters; by = c -> first(c.sites))
        slot = findfirst(e -> e < first(c.sites), layer_ends)
        if isnothing(slot)
            push!(layers, eltype(clusters)[])
            push!(layer_ends, 0)
            slot = length(layers)
        end
        push!(layers[slot], c)
        layer_ends[slot] = last(c.sites)
    end
    return layers
end

"""
    tebd_layers(clusters::Vector{<:ClusterTerm}, dt::Number, alg::TEBD; imaginary_evolution::Bool = false)
    tebd_layers(H::FiniteMPOHamiltonian, dt::Number, alg::TEBD; imaginary_evolution::Bool = false)

Build the Trotter layers used by [`TEBD`](@ref): partition `clusters` (or the [`ClusterTerm`](@ref)s
of `H`) into groups of terms with mutually non-overlapping `sites` ranges, then exponentiate every
term into a gate (returned as a [`ClusterTerm`](@ref) over the same `sites`, whose `op` now holds
`exp(δ * term.op)` instead of `term.op`).

For `order = 2` (the default of [`TEBD`](@ref)), every layer except the last is given a half step
`δ/2` and the last layer a full step `δ`, following the standard palindromic (Strang) composition
`layer₁(δ/2) ⋯ layer_{m-1}(δ/2) layer_m(δ) layer_{m-1}(δ/2) ⋯ layer₁(δ/2)`: `timestep!` applies
every layer once forward (which ends on the full-strength last layer), then every layer except the
last once more in reverse. Splitting the last layer's step in two instead would apply it twice in a
row at `δ/2` with an avoidable truncation in between them, for no accuracy benefit.

The result can be passed directly as the second argument to `timestep!`/`timestep` to skip
recomputing it on repeated calls with the same `H` and `dt`.
"""
function tebd_layers(
        clusters::Vector{<:ClusterTerm}, dt::Number, alg::TEBD;
        imaginary_evolution::Bool = false
    )
    layers = _partition_layers(clusters)
    δ = imaginary_evolution ? -dt : -im * dt
    return map(enumerate(layers)) do (i, layer)
        δᵢ = (alg.order == 2 && i != length(layers)) ? δ / 2 : δ
        return [ClusterTerm(c.sites, exp(scale(c.op, δᵢ))) for c in layer]
    end
end
function tebd_layers(H::FiniteMPOHamiltonian, dt::Number, alg::TEBD; kwargs...)
    return tebd_layers(cluster_hamiltonians(H), dt, alg; kwargs...)
end
