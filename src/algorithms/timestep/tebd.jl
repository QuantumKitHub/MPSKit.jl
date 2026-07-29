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
    _TEBDGate{O}

A single exponentiated [`ClusterTerm`](@ref), pre-decomposed into a `FiniteMPO` (via the same
`decompose_localmpo`/`add_util_leg` route `FiniteMPO(::AbstractTensorMap)` already uses). Building
this once per gate at [`tebd_layers`](@ref) construction time (rather than inside `_apply_gate!`)
means the SVD-based decomposition isn't redone on every single gate application across a whole
`time_evolve` run that reuses the same precomputed `layers`.
"""
struct _TEBDGate{O}
    sites::UnitRange{Int}
    mpo::FiniteMPO{O}
end

"""
    tebd_layers(clusters::Vector{<:ClusterTerm}, dt::Number, alg::TEBD; imaginary_evolution::Bool = false)
    tebd_layers(H::FiniteMPOHamiltonian, dt::Number, alg::TEBD; imaginary_evolution::Bool = false)

Build the Trotter layers used by [`TEBD`](@ref): partition `clusters` (or the [`ClusterTerm`](@ref)s
of `H`) into groups of terms with mutually non-overlapping `sites` ranges, then exponentiate every
term into a gate and decompose it into a [`_TEBDGate`](@ref) over the same `sites`.

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
        return [_TEBDGate(c.sites, FiniteMPO(exp(scale(c.op, δᵢ)))) for c in layer]
    end
end
function tebd_layers(H::FiniteMPOHamiltonian, dt::Number, alg::TEBD; kwargs...)
    return tebd_layers(cluster_hamiltonians(H), dt, alg; kwargs...)
end

# Apply `gate` to `ψ`, evolving `gate.sites` and truncating via `alg_gauge`. Two passes: first fuse
# `gate.mpo` onto `ψ` site by site with `_fuse_mpo_mps` (as `Base.:*(::FiniteMPO,::FiniteMPS)` does),
# splitting off each `AL` losslessly (carrying the leftover bond `C` into the next site's fusion) so
# the whole range is assembled exactly; only then sweep back right-to-left and truncate every bond
# with `alg_gauge`.
function _apply_gate!(ψ::FiniteMPS, gate::_TEBDGate, alg_gauge; normalize::Bool = false)
    start, stop = first(gate.sites), last(gate.sites)
    mpo = gate.mpo
    ψ.AC[start]  # fixes ψ's gauge center at `start`, so ψ.AC/ψ.AR below resolve correctly

    T = TensorOperations.promote_contract(scalartype(mpo), scalartype(ψ))
    A = TensorKit.similarstoragetype(eltype(ψ), T)

    # phase 1: fuse the whole gate onto ψ, losslessly
    Fᵣ = fuser(A, left_virtualspace(ψ, start), left_virtualspace(mpo, 1))
    C_prev = nothing
    for (offset, site) in enumerate(start:stop)
        A1 = site == start ? ψ.AC[site] : ψ.AR[site]
        Fₗ = Fᵣ
        Fᵣ = fuser(A, right_virtualspace(ψ, site), right_virtualspace(mpo, offset))
        fused = _fuse_mpo_mps(mpo[offset], A1, Fₗ, Fᵣ)
        actual = isnothing(C_prev) ? fused : _mul_front(C_prev, fused)
        if site == stop
            ψ.AC[site] = actual
        else
            AL, C, = left_orth(actual)
            ψ.AC[site] = (AL, C)
            C_prev = C
        end
    end

    # phase 2: truncate every bond the gate touched, now that ψ exactly holds the full gate result
    ϵ = zero(real(scalartype(ψ)))
    for site in reverse((start + 1):stop)
        C, AR, ϵᵢ = right_gauge(ψ.AC[site], alg_gauge)
        normalize && normalize!(C)
        ψ.AC[site] = (C, AR)
        ϵ = max(ϵ, ϵᵢ)
    end

    return ψ, ϵ
end

"""
    timestep!(ψ₀::FiniteMPS, H, t, dt, alg::TEBD, [envs]; kwargs...) -> (ψ₀, envs)
    timestep(ψ₀::FiniteMPS, H, t, dt, alg::TEBD, [envs]; kwargs...) -> (ψ, envs)

Time-step `ψ₀` by `dt` using [`TEBD`](@ref). `H` may be a `FiniteMPOHamiltonian`, a
`Vector{<:ClusterTerm}` (as returned by [`cluster_hamiltonians`](@ref)), or precomputed Trotter
layers (as returned by [`tebd_layers`](@ref)) — each tier derives the next-cheapest-to-reuse
representation if it wasn't supplied directly, so a caller who wants to skip recomputing the
Jordan-trace decomposition or the exponentiated gates across repeated calls can pass one of these
in directly instead of `H`.

`envs` is never read: TEBD's update is purely local. It is accepted only for signature parity with
the shared `time_evolve`/`timestep!` dispatch, and threaded through unchanged.
"""
function timestep!(
        ψ::FiniteMPS, H::FiniteMPOHamiltonian, t::Number, dt::Number, alg::TEBD,
        envs = nothing; imaginary_evolution::Bool = false
    )
    clusters = cluster_hamiltonians(H)
    return timestep!(ψ, clusters, t, dt, alg, envs; imaginary_evolution)
end
function timestep!(
        ψ::FiniteMPS, clusters::Vector{<:ClusterTerm}, t::Number, dt::Number, alg::TEBD,
        envs = nothing; imaginary_evolution::Bool = false
    )
    layers = tebd_layers(clusters, dt, alg; imaginary_evolution)
    return timestep!(ψ, layers, t, dt, alg, envs; imaginary_evolution)
end
function timestep!(
        ψ::FiniteMPS, layers::Vector{<:Vector{<:_TEBDGate}}, t::Number, dt::Number, alg::TEBD,
        envs = nothing; imaginary_evolution::Bool = false
    )
    if scalartype(ψ) <: Real && (!imaginary_evolution || !isreal(dt))
        return timestep!(complex(ψ), layers, t, dt, alg, envs; imaginary_evolution)
    end

    # forward pass over every layer (ends on the full-strength last layer for order = 2)
    for layer in layers, gate in layer
        _apply_gate!(ψ, gate, alg.alg_gauge; normalize = imaginary_evolution)
    end

    # order = 2: backward pass over every layer except the last (already applied at full strength)
    if alg.order == 2
        for layer in reverse(layers[1:(end - 1)]), gate in layer
            _apply_gate!(ψ, gate, alg.alg_gauge; normalize = imaginary_evolution)
        end
    end

    return ψ, envs
end

# copying version: works for any of the three input tiers (H, clusters, or layers) since the
# element type of the second argument is left generic here and resolved by the `timestep!` methods
function timestep(
        ψ::FiniteMPS, H, t::Number, dt::Number, alg::TEBD, envs = nothing;
        imaginary_evolution::Bool = false
    )
    ψ′ = (scalartype(ψ) <: Real && !imaginary_evolution) ? complex(ψ) : copy(ψ)
    return timestep!(ψ′, H, t, dt, alg, envs; imaginary_evolution)
end
