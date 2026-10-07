"""
$(TYPEDEF)

Algorithm for constructing a finite or infinite time evolution MPO using the nonperturbative cluster expansion.
Exact exponentials are matched on clusters of up to `N` sites.
For nearest-neighbor Hamiltonians the error is `O(dt^N)`.

# Fields

$(TYPEDFIELDS)

# Algorithm

Exact evolution within clusters of up to `N` sites is included to all orders in
`dt`, together with products of disjoint clusters. The remaining error comes from
connected contributions extending beyond `N` sites: for nearest-neighbor
Hamiltonians the one-step error is `O(dt^N)`. Infinite Hamiltonians include clusters
crossing the unit-cell boundary; finite chains become exact up to roundoff when
`N` reaches the chain length.

Compared with [`TaylorCluster`](@ref), which matches a specified Taylor order,
`ClusterExpansion` controls the spatial extent of the exact evolution. Its `N`
is a cluster size, not a Taylor order. Resumming smaller clusters can reduce the
error prefactor and permit larger time steps. Compared with the compact, local
construction of [`WII`](@ref), increasing `N` systematically incorporates larger
clusters, at the cost of exponentially growing cluster exponentials and MPO bond
dimensions. For full-rank uniform physical spaces of dimension `d`, the latter
scale as `1 + d^2 + ⋯ + d^(2 floor(N/2))`; mixed spaces or rank-deficient residuals
can require extra channels.

This is useful for evolution with fewer, larger steps and for growing the MPS bond
dimension from weakly entangled states. Larger steps can also increase compression
costs, so check convergence and runtime by varying `dt` and `N`. MPS compression
introduces a separate error, and the approximate MPO need not be unitary. `tol`
only identifies shared SVD support: smaller singular components are retained in
complementary channels rather than discarded.

# See also

Used as the `algorithm` argument of [`make_time_mpo`](@ref).

# References

* [Vanhecke et al. Phys. Rev. A 103 (2021)](@cite vanhecke2021cluster)
"""
struct ClusterExpansion <: Algorithm
    "maximum cluster size (positive integer, default: `2`)"
    N::Int
    "relative singular-value cutoff for shared SVD support (`0 < tol < 1`, default: `1.0e-12`)"
    tol::Float64
    function ClusterExpansion(N::Integer; tol::Real = 1.0e-12)
        N ≥ 1 || throw(ArgumentError("cluster size must be positive"))
        isfinite(tol) && 0 < tol < 1 || throw(ArgumentError("tol must be between zero and one"))
        return new(N, tol)
    end
end
ClusterExpansion(; N::Integer = 2, kwargs...) = ClusterExpansion(N; kwargs...)

function make_time_mpo(H::MPOHamiltonian, dt::Number, alg::ClusterExpansion)
    # The only dispatch on a runtime cluster size. All subsequent stages and
    # contraction shapes are determined by this Val parameter.
    isempty(H) && throw(ArgumentError("Hamiltonian must contain at least one site"))
    N = isfinite(H) ? min(alg.N, length(H)) : alg.N
    return make_cluster_expansion_mpo(H, -im * dt, Val(N), alg.tol)
end

function make_cluster_expansion_mpo(H::MPOHamiltonian, τ::Number, n::Val, tol::Real)
    T = promote_type(scalartype(H), typeof(τ))
    storage = TensorKit.similarstoragetype(storagetype(H), T)
    O = map(parent(H)) do h
        return exponentiate_onsite_tensor(h, τ, storage)
    end
    # These cases are already exact products; avoid building channels from
    # roundoff in differences of independently contracted exponentials.
    (iszero(τ) || all(has_only_onsite_terms, parent(H))) && return assemble_evolution_mpo(O)
    return assemble_evolution_mpo(add_cluster_corrections(O, H, τ, n, tol))
end

has_only_onsite_terms(h) = size(h, 1) ≤ 2 && size(h, 4) ≤ 2

assemble_evolution_mpo(O::PeriodicVector) = InfiniteMPO(O)
function assemble_evolution_mpo(O::Vector)
    O[1] = O[1][1, :, :, :]
    O[end] = O[end][:, :, :, 1]
    # Remove structurally unused levels only after all stages have finished.
    return remove_orphans!(FiniteMPO(O); tol = 0)
end

function exponentiate_onsite_tensor(h, τ, ::Type{A}) where {A}
    P = physicalspace(h)
    levels = ⊞(oneunit(P))
    blocktype = tensormaptype(spacetype(P), 2, 2, A)
    O = SparseBlockTensorMap{blocktype}(undef, levels ⊗ P ← P ⊗ levels)
    onsite = removeunit(removeunit(h[1, 1, 1, end], 4), 1)
    O[1, 1, 1, 1] = add_util_leg(exp(τ * onsite))
    return O
end

add_cluster_corrections(O, H, τ, ::Val{1}, tol) = O
function add_cluster_corrections(O, H, τ, n::Val{N}, tol) where {N}
    previous = add_cluster_corrections(O, H, τ, Val(N - 1), tol)
    starts = isfinite(H) ? (1:(length(H) - N + 1)) : (1:length(H))
    nt = (N - 1) ÷ 2
    level = N ÷ 2 + 1
    # Every translated window must see the completed previous stage. Compute
    # all corrections before inserting any of them, including across the cell seam.
    if isodd(N)
        centers = [solve_center_correction(previous, s, evolution_cluster_residual(H, previous, s, τ, n), n) for s in starts]
        for s in starts
            previous[s + nt][level, 1, 1, level] = centers[s]
        end
        return previous
    else
        pairs = [factor_two_site_correction(H, previous, s, τ, n, tol) for s in starts]
        expanded = expand_correction_bonds(previous, pairs, n)
        for s in starts
            expanded[s + nt][level - 1, 1, 1, level] = pairs[s][1]
            expanded[s + nt + 1][level, 1, 1, level - 1] = pairs[s][2]
        end
        return expanded
    end
end

function expand_correction_bonds(O::PeriodicVector, pairs, ::Val{N}) where {N}
    nt = (N - 1) ÷ 2
    bonds = PeriodicArray([right_virtualspace(pairs[mod1(i - nt, length(O))][1]) for i in 1:length(O)])
    return PeriodicArray([append_virtual_spaces(O[i], bonds[i - 1], bonds[i]) for i in 1:length(O)])
end

function expand_correction_bonds(O::Vector, pairs, ::Val{N}) where {N}
    nt = (N - 1) ÷ 2
    empty_bond = zero(left_virtualspace(O[1])[1])
    # Only the middle bonds can support an N-site cluster. Empty levels keep
    # the same level indices everywhere until the final boundary projection.
    bonds = vcat(
        fill(empty_bond, nt + 1),
        [right_virtualspace(pair[1]) for pair in pairs],
        fill(empty_bond, nt + 1)
    )
    return [append_virtual_spaces(O[i], bonds[i], bonds[i + 1]) for i in 1:length(O)]
end

function append_virtual_spaces(O, left_bond, right_bond)
    L, R = left_virtualspace(O), right_virtualspace(O)
    left_levels = L ⊞ left_bond
    right_levels = R ⊞ right_bond
    P = physicalspace(O)
    expanded = typeof(O)(undef, left_levels ⊗ P ← P ⊗ right_levels)
    for (indices, block) in nonzero_pairs(O)
        expanded[indices] = block
    end
    return expanded
end

function evolution_cluster_residual(H, O, start::Int, τ, n::Val{N}) where {N}
    boundary = size(H[start + N - 1], 4)
    exact = exp(τ * contract_mpo_window(H, start, n, boundary))
    return add_util_leg(exact - contract_mpo_window(O, start, n, 1))
end

function factor_two_site_correction(H, O, start, τ, n::Val{2}, tol)
    residual = evolution_cluster_residual(H, O, start, τ, n)
    A, B = factor_with_complementary_spaces(permute(residual, ((1, 2, 4), (3, 5, 6))), tol)
    return permute(A, ((1, 2), (3, 4))), permute(B, ((1, 2), (3, 4)))
end

function factor_two_site_correction(H, O, start, τ, n::Val{N}, tol) where {N}
    residual = evolution_cluster_residual(H, O, start, τ, n)
    left_map, right_map = correction_environment_maps(O, start, n)
    center = solve_center_correction(left_map, right_map, residual, n)
    C = permute(center, ((1, 2, 4), (3, 5, 6)))

    # The minimum-norm solves restrict the center to the active environment
    # subspaces. Complete only these directions, rather than the redundant
    # virtual directions introduced by previous stages. Untruncated LQ/QR bases
    # retain the entire physical environment support, including weak sectors.
    _, Q_left = right_orth(left_map)
    Q_right, _ = left_orth(right_map)
    F_left = Q_left' ⊗ id(storagetype(C), codomain(C)[2] ⊗ codomain(C)[3])
    F_right = id(storagetype(C), domain(C)[1] ⊗ domain(C)[2]) ⊗ Q_right
    A, B = factor_with_complementary_spaces(F_left' * C * F_right, tol)
    return permute(F_left * A, ((1, 2), (3, 4))),
        permute(B * F_right', ((1, 2), (3, 4)))
end

"""
    factor_with_complementary_spaces(C, tol)

Factor `C` through its shared SVD support and separate left/right complementary
spaces. For exact rank `r_c`, the virtual multiplicity in sector `c` is
`m_c + n_c - r_c`, the minimum that gives both factors full environment support.
For exact null directions, the factors have the form

    A = [U√S  γU⊥  0],    B = [√S Vᴴ; 0; γV⊥ᴴ].

At a finite rank cutoff, retain the remaining block `E = U⊥' C V⊥` in the
complementary channels, so the product still reconstructs `C` up to roundoff.
The scale `γ` balances these channels against the retained singular values.
"""
function factor_with_complementary_spaces(C::TensorMap, tol::Real)
    cnorm = norm(C)
    if iszero(cnorm)
        # At zero rank, the two complementary spaces occupy disjoint channels.
        L, R = fuse(codomain(C)), fuse(domain(C))
        storage = storagetype(C)
        A = catdomain(
            isomorphism(storage, codomain(C) ← L), zeros(storage, codomain(C) ← R)
        )
        B = catcodomain(
            zeros(storage, L ← domain(C)), isomorphism(storage, R ← domain(C))
        )
        return A, B
    end
    U, S, Vᴴ, _ = svd_trunc(C; trunc = trunctol(; rtol = tol, p = Inf))
    U_perp, V_perp = left_null(U), right_null(Vᴴ)
    E = U_perp' * C * V_perp'
    γ = sqrt(cnorm)
    root = sqrt(S)
    A = catdomain(catdomain(U * root, γ * U_perp), U_perp * E / (2γ))
    B = catcodomain(catcodomain(root * Vᴴ, E * V_perp / (2γ)), γ * V_perp)
    return A, B
end

function contract_mpo_window(O, start::Int, ::Val{1}, boundary::Int)
    return removeunit(removeunit(TensorMap(O[start][1, :, :, boundary]), 4), 1)
end

function contract_mpo_window(O, start::Int, n::Val{N}, boundary::Int) where {N}
    left = removeunit(TensorMap(O[start][1, :, :, :]), 1)
    right = removeunit(TensorMap(O[start + N - 1][:, :, :, boundary]), 4)
    sites = collect_bulk_tensors(O, start + 1, Val(N - 2))
    return contract_mpo_window(left, sites, right, n)
end

collect_bulk_tensors(O, start, ::Val{0}) = ()
function collect_bulk_tensors(O, start, ::Val{N}) where {N}
    return (TensorMap(O[start]), collect_bulk_tensors(O, start + 1, Val(N - 1))...)
end

"""
    contract_mpo_window(left, sites, right, ::Val{N})

Contract the open-boundary cluster, shown for `N = 4`. `L` and `R` are the
boundary tensors; `O₂` and `O₃` are the site-dependent bulk tensors. Negative labels are open
physical legs and positive labels are contracted virtual bonds. Domain legs
are above the tensors and codomain legs below.

```
 -5      -6      -7      -8
  │       │       │       │
┌─┴─┐   ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
│ L ├─1─┤O₂ ├─2─┤O₃ ├─3─┤ R │
└─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
  │       │       │       │
 -1      -2      -3      -4
```
"""
@generated function contract_mpo_window(
        left::AbstractTensorMap{<:Any, S, 1, 2}, sites::NTuple{K, AbstractTensorMap{<:Any, S, 2, 2}},
        right::AbstractTensorMap{<:Any, S, 2, 1}, ::Val{N}
    ) where {S, N, K}
    out = tensorexpr(:cluster, -(1:N), -((N + 1):(2N)))
    first_site = tensorexpr(:left, -1, (-N - 1, 1))
    last_site = tensorexpr(:right, (N - 1, -N), -2N)
    middle = [tensorexpr(:(sites[$(i - 1)]), (i - 1, -i), (-N - i, i)) for i in 2:(N - 1)]
    return macroexpand(@__MODULE__, :(return @plansor $out := *($first_site, $(middle...), $last_site)))
end

left_correction_environment(O, start, ::Val{1}) = O[start][1, 1, 1, 2]
function left_correction_environment(O, start, ::Val{L}) where {L}
    left = left_correction_environment(O, start, Val(L - 1))
    return extend_left_environment(left, O[start + L - 1][L, 1, 1, L + 1])
end

right_correction_environment(O, start, ::Val{1}) = O[start][2, 1, 1, 1]
function right_correction_environment(O, start, ::Val{L}) where {L}
    right = right_correction_environment(O, start + 1, Val(L - 1))
    return extend_right_environment(O[start][L + 1, 1, 1, L], right)
end

"""
    extend_left_environment(left, site)

Append a site to the left environment, shown for `K = numout(left) = 2`.
For general `K`, each vertical leg of `left` is a bundle of `K - 1` physical
legs. The outer virtual legs remain open and bond `1` is contracted. Physical
domain legs are above the tensors and codomain legs below.

```
       -4        -5
        │         │
     ┌──┴──┐   ┌──┴──┐
-1───┤left ├─1─┤site ├───-6
     └──┬──┘   └──┬──┘
        │         │
       -2        -3
```
"""
@generated function extend_left_environment(left::AbstractTensorMap{<:Any, <:Any, K, K}, site::AbstractTensorMap) where {K}
    out = tensorexpr(:next_left, -(1:(K + 1)), -((K + 2):(2K + 2)))
    env = tensorexpr(:left, -(1:K), (-((K + 2):(2K))..., 1))
    op = tensorexpr(:site, (1, -K - 1), (-2K - 1, -2K - 2))
    return macroexpand(@__MODULE__, :(return @plansor $out := $env * $op))
end

"""
    extend_right_environment(site, right)

Prepend a site to the right environment, shown for `K = numout(right) = 2`.
For general `K`, each vertical leg of `right` is a bundle of `K - 1` physical
legs. The outer virtual legs remain open and bond `1` is contracted. Physical
domain legs are above the tensors and codomain legs below.

```
       -4        -5
        │         │
     ┌──┴──┐   ┌──┴──┐
-1───┤site ├─1─┤right├───-6
     └──┬──┘   └──┬──┘
        │         │
       -2        -3
```
"""
@generated function extend_right_environment(site::AbstractTensorMap, right::AbstractTensorMap{<:Any, <:Any, K, K}) where {K}
    out = tensorexpr(:next_right, -(1:(K + 1)), -((K + 2):(2K + 2)))
    op = tensorexpr(:site, (-1, -2), (-K - 2, 1))
    env = tensorexpr(:right, (1, -(3:(K + 1))...), -((K + 3):(2K + 2)))
    return macroexpand(@__MODULE__, :(return @plansor $out := $op * $env))
end

function correction_environment_maps(O, start, ::Val{N}) where {N}
    L = (N - 1) ÷ 2
    left = left_correction_environment(O, start, Val(L))
    right = right_correction_environment(O, start + N - L, Val(L))
    indices = ntuple(identity, Val(2L + 1))
    return permute(left, (indices, (2L + 2,))),
        permute(right, ((1,), indices .+ 1))
end

function solve_center_correction(O, start::Int, residual::TensorMap, n::Val{N}) where {N}
    left_map, right_map = correction_environment_maps(O, start, n)
    return solve_center_correction(left_map, right_map, residual, n)
end

"""
    solve_center_correction(left_map, right_map, residual, ::Val{N})

Solve the environment equations from the left and right without forming inverse
maps. Completed environments have full row/column support in each symmetry
sector; rectangular solves choose the minimum-norm center.

The grouped map layouts for `N = 3` are shown below. Domain bundles are above
and codomain bundles below. `pᵢ′` is the dual physical space, and `u_L`/`u_R`
are the trivial boundary spaces.

First group the left physical legs of the residual into `B_L` and solve `L \\ B_L`:

```
          V_L                 (p₂′, p₃′, p₂, p₃, u_R)
           │                             │
     ┌─────┴─────┐                 ┌─────┴─────┐
     │     L     │                 │    B_L    │
     └─────┬─────┘                 └─────┬─────┘
           │                             │
    (u_L, p₁, p₁′)                (u_L, p₁, p₁′)
```

Regroup that solution into `B_R` and solve `B_R / R`:

```
     (p₃′, p₃, u_R)                (p₃′, p₃, u_R)
           │                             │
     ┌─────┴─────┐                 ┌─────┴─────┐
     │     R     │                 │    B_R    │
     └─────┬─────┘                 └─────┬─────┘
           │                             │
          V_R                     (V_L, p₂, p₂′)
```

Finally move `p₂′` back into the domain. The center has one physical leg per
side for odd `N`, or two for even `N`, and its two virtual legs. For larger
clusters, each outer physical space above is a bundle of `(N - 1) ÷ 2` legs.
"""
@generated function solve_center_correction(
        left_map::TensorMap, right_map::TensorMap, residual::TensorMap, ::Val{N}
    ) where {N}
    nt = (N - 1) ÷ 2
    width = N - 2nt
    # Group all left environment legs into the codomain for the left solve.
    left_indices = (1, (2:(nt + 1))..., ((N + 2):(N + nt + 1))...)
    remaining = (((nt + 2):(N + 1))..., ((N + nt + 2):(2N + 2))...)
    # After solving, group the right environment legs into the domain.
    middle_indices = (1, (2:(width + 1))..., ((nt + width + 2):(nt + 2width + 1))...)
    right_indices = (((width + 2):(nt + width + 1))..., ((nt + 2width + 2):(2nt + 2width + 2))...)
    center_indices = (Tuple(1:(width + 1)), Tuple((width + 2):(2width + 2)))
    return quote
        left_rhs = permute(residual, $((left_indices, remaining)))
        left_solved = left_map \ left_rhs
        right_rhs = permute(left_solved, $((middle_indices, right_indices)))
        solved = right_rhs / right_map
        return permute(solved, $center_indices)
    end
end
