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

function make_cluster_expansion_mpo(H::MPOHamiltonian, τ::Number, ::Val{N}, tol::Real) where {N}
    storage_type = TensorKit.similarstoragetype(storagetype(H), promote_type(scalartype(H), typeof(τ)))
    return make_cluster_expansion_mpo(H, τ, Val(N), tol, storage_type)
end
function make_cluster_expansion_mpo(
        H::MPOHamiltonian, τ::Number, ::Val{N}, tol::Real, ::Type{A}
    ) where {N, A}
    tensors = map(parent(H)) do ham_tensor
        return exponentiate_onsite_tensor(ham_tensor, τ, A)
    end
    # These cases are already exact products; avoid building channels from
    # roundoff in differences of independently contracted exponentials.
    if !(iszero(τ) || all(has_only_onsite_terms, parent(H)))
        tensors = add_cluster_corrections(tensors, H, τ, Val(N), tol)
    end

    return assemble_evolution_mpo(tensors)
end

has_only_onsite_terms(ham_tensor) = size(ham_tensor, 1) ≤ 2 && size(ham_tensor, 4) ≤ 2

assemble_evolution_mpo(tensors::PeriodicVector) = InfiniteMPO(tensors)
function assemble_evolution_mpo(tensors::Vector)
    tensors[1] = tensors[1][1, :, :, :]
    tensors[end] = tensors[end][:, :, :, 1]
    # Remove structurally unused levels only after all stages have finished.
    return remove_orphans!(FiniteMPO(tensors); tol = 0)
end

function exponentiate_onsite_tensor(ham_tensor, τ, ::Type{A}) where {A}
    physical_space = physicalspace(ham_tensor)
    levels = ⊞(oneunit(physical_space))
    blocktype = tensormaptype(spacetype(physical_space), 2, 2, A)
    tensor = SparseBlockTensorMap{blocktype}(undef, levels ⊗ physical_space ← physical_space ⊗ levels)
    onsite = removeunit(removeunit(ham_tensor[1, 1, 1, end], 4), 1)
    tensor[1, 1, 1, 1] = add_util_leg(exp(τ * onsite))
    return tensor
end

function add_cluster_corrections(tensors, H, τ, ::Val{N}, tol) where {N}
    N == 1 && return tensors
    previous = add_cluster_corrections(tensors, H, τ, Val(N - 1), tol)
    starts = isfinite(H) ? (1:(length(H) - N + 1)) : (1:length(H))
    environment_length = (N - 1) ÷ 2
    level = N ÷ 2 + 1
    # Every translated window must see the completed previous stage. Compute
    # all corrections before inserting any of them, including across the cell seam.
    if isodd(N)
        centers = [solve_center_correction(previous, start, evolution_cluster_residual(H, previous, start, τ, Val(N)), Val(N)) for start in starts]
        for start in starts
            previous[start + environment_length][level, 1, 1, level] = centers[start]
        end
        return previous
    else
        factors = [factor_two_site_correction(H, previous, start, τ, Val(N), tol) for start in starts]
        expanded = expand_correction_bonds(previous, factors, Val(N))
        for start in starts
            expanded[start + environment_length][level - 1, 1, 1, level] = factors[start][1]
            expanded[start + environment_length + 1][level, 1, 1, level - 1] = factors[start][2]
        end
        return expanded
    end
end

function expand_correction_bonds(tensors::PeriodicVector, factors, ::Val{N}) where {N}
    environment_length = (N - 1) ÷ 2
    bonds = PeriodicArray([right_virtualspace(factors[mod1(i - environment_length, length(tensors))][1]) for i in 1:length(tensors)])
    return PeriodicArray([append_virtual_spaces(tensors[i], bonds[i - 1], bonds[i]) for i in 1:length(tensors)])
end

function expand_correction_bonds(tensors::Vector, factors, ::Val{N}) where {N}
    environment_length = (N - 1) ÷ 2
    empty_bond = zero(left_virtualspace(tensors[1])[1])
    # Only the middle bonds can support an N-site cluster. Empty levels keep
    # the same level indices everywhere until the final boundary projection.
    bonds = vcat(
        fill(empty_bond, environment_length + 1),
        [right_virtualspace(factor[1]) for factor in factors],
        fill(empty_bond, environment_length + 1)
    )
    return [append_virtual_spaces(tensors[i], bonds[i], bonds[i + 1]) for i in 1:length(tensors)]
end

function append_virtual_spaces(tensor, left_bond, right_bond)
    left_levels = left_virtualspace(tensor) ⊞ left_bond
    right_levels = right_virtualspace(tensor) ⊞ right_bond
    physical_space = physicalspace(tensor)
    expanded = typeof(tensor)(undef, left_levels ⊗ physical_space ← physical_space ⊗ right_levels)
    for (indices, block) in nonzero_pairs(tensor)
        expanded[indices] = block
    end
    return expanded
end

function evolution_cluster_residual(H, tensors, start::Int, τ, ::Val{N}) where {N}
    boundary = size(H[start + N - 1], 4)
    exact = exp(τ * contract_mpo_window(H, start, Val(N), boundary))
    return add_util_leg(exact - contract_mpo_window(tensors, start, Val(N), 1))
end

function factor_two_site_correction(H, tensors, start, τ, ::Val{N}, tol) where {N}
    residual = evolution_cluster_residual(H, tensors, start, τ, Val(N))
    if N == 2
        left_factor, right_factor = factor_with_complementary_spaces(permute(residual, ((1, 2, 4), (3, 5, 6))), tol)
        return permute(left_factor, ((1, 2), (3, 4))), permute(right_factor, ((1, 2), (3, 4)))
    end
    left_map, right_map = correction_environment_maps(tensors, start, Val(N))
    center = solve_center_correction(left_map, right_map, residual, Val(N))
    center_map = permute(center, ((1, 2, 4), (3, 5, 6)))

    # The minimum-norm solves restrict the center to the active environment
    # subspaces. Complete only these directions, rather than the redundant
    # virtual directions introduced by previous stages. Untruncated LQ/QR bases
    # retain the entire physical environment support, including weak sectors.
    _, left_basis = right_orth(left_map)
    right_basis, _ = left_orth(right_map)
    left_embedding = left_basis' ⊗ id(storagetype(center_map), codomain(center_map)[2] ⊗ codomain(center_map)[3])
    right_embedding = id(storagetype(center_map), domain(center_map)[1] ⊗ domain(center_map)[2]) ⊗ right_basis
    left_factor, right_factor = factor_with_complementary_spaces(left_embedding' * center_map * right_embedding, tol)
    return permute(left_embedding * left_factor, ((1, 2), (3, 4))),
        permute(right_factor * right_embedding', ((1, 2), (3, 4)))
end

"""
    factor_with_complementary_spaces(tensor, tol)

Factor `tensor` through its shared SVD support and separate left/right complementary
spaces. For exact rank `r_c`, the virtual multiplicity in sector `c` is
`m_c + n_c - r_c`, the minimum that gives both factors full environment support.
For exact null directions, the factors have the form

    A = [U√S  γU⊥  0],    B = [√S Vᴴ; 0; γV⊥ᴴ].

At a finite rank cutoff, retain the remaining block in the complementary
channels, so the product still reconstructs `tensor` up to roundoff.
The scale `γ` balances these channels against the retained singular values.
"""
function factor_with_complementary_spaces(tensor::TensorMap, tol::Real)
    tensor_norm = norm(tensor)
    if iszero(tensor_norm)
        # At zero rank, the two complementary spaces occupy disjoint channels.
        left_space, right_space = fuse(codomain(tensor)), fuse(domain(tensor))
        storage_type = storagetype(tensor)
        left_factor = catdomain(
            isomorphism(storage_type, codomain(tensor) ← left_space), zeros(storage_type, codomain(tensor) ← right_space)
        )
        right_factor = catcodomain(
            zeros(storage_type, left_space ← domain(tensor)), isomorphism(storage_type, right_space ← domain(tensor))
        )
        return left_factor, right_factor
    end
    U, S, Vᴴ, _ = svd_trunc(tensor; trunc = trunctol(; rtol = tol, p = Inf))
    U_perp, V_perp = left_null(U), right_null(Vᴴ)
    complement = U_perp' * tensor * V_perp'
    complement_scale = sqrt(tensor_norm)
    sqrt_values = sqrt(S)
    left_factor = catdomain(catdomain(U * sqrt_values, complement_scale * U_perp), U_perp * complement / (2complement_scale))
    right_factor = catcodomain(catcodomain(sqrt_values * Vᴴ, complement * V_perp / (2complement_scale)), complement_scale * V_perp)
    return left_factor, right_factor
end

function contract_mpo_window(tensors, start::Int, ::Val{N}, boundary::Int) where {N}
    N == 1 && return removeunit(removeunit(TensorMap(tensors[start][1, :, :, boundary]), 4), 1)
    left = removeunit(TensorMap(tensors[start][1, :, :, :]), 1)
    right = removeunit(TensorMap(tensors[start + N - 1][:, :, :, boundary]), 4)
    sites = collect_bulk_tensors(tensors, start + 1, Val(N - 2))
    return contract_mpo_window(left, sites, right, Val(N))
end

function collect_bulk_tensors(tensors, start, ::Val{N}) where {N}
    N == 0 && return ()
    return (TensorMap(tensors[start]), collect_bulk_tensors(tensors, start + 1, Val(N - 1))...)
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
    result = tensorexpr(:cluster, -(1:N), -((N + 1):(2N)))
    first_site = tensorexpr(:left, -1, (-N - 1, 1))
    last_site = tensorexpr(:right, (N - 1, -N), -2N)
    bulk = [tensorexpr(:(sites[$(i - 1)]), (i - 1, -i), (-N - i, i)) for i in 2:(N - 1)]
    return macroexpand(@__MODULE__, :(return @plansor $result := *($first_site, $(bulk...), $last_site)))
end

function left_correction_environment(tensors, start, ::Val{N}) where {N}
    N == 1 && return tensors[start][1, 1, 1, 2]
    left = left_correction_environment(tensors, start, Val(N - 1))
    return extend_left_environment(left, tensors[start + N - 1][N, 1, 1, N + 1])
end

function right_correction_environment(tensors, start, ::Val{N}) where {N}
    N == 1 && return tensors[start][2, 1, 1, 1]
    right = right_correction_environment(tensors, start + 1, Val(N - 1))
    return extend_right_environment(tensors[start][N + 1, 1, 1, N], right)
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
    result = tensorexpr(:next_left, -(1:(K + 1)), -((K + 2):(2K + 2)))
    environment = tensorexpr(:left, -(1:K), (-((K + 2):(2K))..., 1))
    site_expression = tensorexpr(:site, (1, -K - 1), (-2K - 1, -2K - 2))
    return macroexpand(@__MODULE__, :(return @plansor $result := $environment * $site_expression))
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
    result = tensorexpr(:next_right, -(1:(K + 1)), -((K + 2):(2K + 2)))
    site_expression = tensorexpr(:site, (-1, -2), (-K - 2, 1))
    environment = tensorexpr(:right, (1, -(3:(K + 1))...), -((K + 3):(2K + 2)))
    return macroexpand(@__MODULE__, :(return @plansor $result := $site_expression * $environment))
end

function correction_environment_maps(tensors, start, ::Val{N}) where {N}
    environment_length = (N - 1) ÷ 2
    left = left_correction_environment(tensors, start, Val(environment_length))
    right = right_correction_environment(tensors, start + N - environment_length, Val(environment_length))
    environment_indices = ntuple(identity, Val(2environment_length + 1))
    return permute(left, (environment_indices, (2environment_length + 2,))), permute(right, ((1,), environment_indices .+ 1))
end

function solve_center_correction(tensors, start::Int, residual::TensorMap, ::Val{N}) where {N}
    left_map, right_map = correction_environment_maps(tensors, start, Val(N))
    return solve_center_correction(left_map, right_map, residual, Val(N))
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
    environment_length = (N - 1) ÷ 2
    center_width = N - 2environment_length
    # Group all left environment legs into the codomain for the left solve.
    left_indices = (1, (2:(environment_length + 1))..., ((N + 2):(N + environment_length + 1))...)
    remaining_indices = (((environment_length + 2):(N + 1))..., ((N + environment_length + 2):(2N + 2))...)
    # After solving, group the right environment legs into the domain.
    middle_indices = (1, (2:(center_width + 1))..., ((environment_length + center_width + 2):(environment_length + 2center_width + 1))...)
    right_indices = (((center_width + 2):(environment_length + center_width + 1))..., ((environment_length + 2center_width + 2):(2environment_length + 2center_width + 2))...)
    center_indices = (Tuple(1:(center_width + 1)), Tuple((center_width + 2):(2center_width + 2)))
    return quote
        left_solved = left_map \ permute(residual, $((left_indices, remaining_indices)))
        solved = permute(left_solved, $((middle_indices, right_indices))) / right_map
        return permute(solved, $center_indices)
    end
end
