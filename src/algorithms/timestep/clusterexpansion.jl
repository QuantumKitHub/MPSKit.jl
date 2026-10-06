"""
    ClusterExpansion(N; tol = 1.0e-12)
    ClusterExpansion(; N = 2, tol = 1.0e-12)

Construct an infinite time evolution MPO for a nearest-neighbor Hamiltonian by
matching exact exponentials on clusters of up to `N` sites. This is the nonperturbative construction of
[Vanhecke, Vanderstraeten and Verstraete (2021)](https://doi.org/10.1103/PhysRevA.103.L020402).
For nearest-neighbor Hamiltonians the error is `O(dt^N)`.

The unit cell may contain different physical spaces of the same TensorKit space type.
Symmetry sectors are preserved throughout the construction. Every translated cluster
is matched, including those crossing the unit-cell boundary.

For full-rank uniform physical spaces of dimension `d`, the virtual dimension grows
as `1 + d^2 + ⋯ + d^(2 floor(N/2))`. Mixed physical spaces and rank-deficient residuals
can require additional complementary virtual channels. Exact cluster exponentials
also grow exponentially with `N`. `tol` is the relative singular-value cutoff used
to identify shared SVD support; components below the cutoff are retained in the
complementary channels. Environment equations are solved directly without a
second singular-value cutoff.
"""
struct ClusterExpansion <: Algorithm
    N::Int
    tol::Float64
    function ClusterExpansion(N::Integer; tol::Real = 1.0e-12)
        N ≥ 1 || throw(ArgumentError("cluster size must be positive"))
        isfinite(tol) && 0 < tol < 1 || throw(ArgumentError("tol must be between zero and one"))
        return new(N, tol)
    end
end
ClusterExpansion(; N::Integer = 2, kwargs...) = ClusterExpansion(N; kwargs...)

function make_time_mpo(H::InfiniteMPOHamiltonian, dt::Number, alg::ClusterExpansion)
    # The only dispatch on a runtime cluster size. All subsequent stages and
    # contraction shapes are determined by this Val parameter.
    return make_cluster_mpo(H, -im * dt, Val(alg.N), alg.tol)
end

function make_cluster_mpo(H::InfiniteMPOHamiltonian, τ::Number, n::Val, tol::Real)
    T = promote_type(scalartype(H), typeof(τ))
    storage = TensorKit.similarstoragetype(storagetype(H), T)
    O = PeriodicArray([cluster_initial_tensor(H[i], τ, storage) for i in 1:length(H)])
    # These cases are already exact products; avoid building channels from
    # roundoff in differences of independently contracted exponentials.
    (iszero(τ) || all(cluster_is_onsite, parent(H))) && return InfiniteMPO(O)
    return InfiniteMPO(cluster_build(O, H, τ, n, tol))
end

cluster_is_onsite(h) = size(h, 1) == 2 && size(h, 4) == 2

function cluster_initial_tensor(h, τ, ::Type{A}) where {A}
    P = physicalspace(h)
    levels = ⊞(oneunit(P))
    blocktype = tensormaptype(spacetype(P), 2, 2, A)
    O = SparseBlockTensorMap{blocktype}(undef, levels ⊗ P ← P ⊗ levels)
    onsite = removeunit(removeunit(h[1, 1, 1, end], 4), 1)
    O[1, 1, 1, 1] = add_util_leg(exp(τ * onsite))
    return O
end

cluster_build(O, H, τ, ::Val{1}, tol) = O
function cluster_build(O, H, τ, n::Val{N}, tol) where {N}
    previous = cluster_build(O, H, τ, Val(N - 1), tol)
    nt = (N - 1) ÷ 2
    level = N ÷ 2 + 1
    # Every translated window must see the completed previous stage. Compute
    # all corrections before inserting any of them, including across the cell seam.
    if isodd(N)
        centers = [cluster_correction(H, previous, s, τ, n) for s in 1:length(H)]
        for s in 1:length(H)
            previous[s + nt][level, 1, 1, level] = centers[s]
        end
        return previous
    else
        pairs = [cluster_pair(H, previous, s, τ, n, tol) for s in 1:length(H)]
        expanded = cluster_expand(previous, pairs, n)
        for s in 1:length(H)
            expanded[s + nt][level - 1, 1, 1, level] = pairs[s][1]
            expanded[s + nt + 1][level, 1, 1, level - 1] = pairs[s][2]
        end
        return expanded
    end
end

function cluster_expand(O, pairs, ::Val{N}) where {N}
    nt = (N - 1) ÷ 2
    bonds = PeriodicArray([right_virtualspace(pairs[mod1(i - nt, length(O))][1]) for i in 1:length(O)])
    return PeriodicArray([cluster_expand_tensor(O[i], bonds[i - 1], bonds[i]) for i in 1:length(O)])
end

function cluster_expand_tensor(O, left_bond, right_bond)
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

function cluster_residual(H, O, start::Int, τ, n::Val{N}) where {N}
    boundary = size(H[start + N - 1], 4)
    exact = exp(τ * cluster_contract(H, start, n, boundary))
    return add_util_leg(exact - cluster_contract(O, start, n, 1))
end

function cluster_correction(H, O, start, τ, n::Val)
    residual = cluster_residual(H, O, start, τ, n)
    return cluster_center(O, start, residual, n)
end

function cluster_pair(H, O, start, τ, n::Val{2}, tol)
    residual = cluster_residual(H, O, start, τ, n)
    return cluster_split(residual, tol)
end

function cluster_pair(H, O, start, τ, n::Val{N}, tol) where {N}
    residual = cluster_residual(H, O, start, τ, n)
    nt = (N - 1) ÷ 2
    left, right = cluster_environments(O, start, n)
    left_map, right_map = cluster_environment_maps(left, right, Val(nt))
    center = cluster_center(left_map, right_map, residual, n)
    C = permute(center, ((1, 2, 4), (3, 5, 6)))

    # The minimum-norm solves restrict the center to the active environment
    # subspaces. Complete only these directions, rather than the redundant
    # virtual directions introduced by previous stages. Untruncated LQ/QR bases
    # retain the entire physical environment support, including weak sectors.
    _, Q_left = right_orth(left_map)
    Q_right, _ = left_orth(right_map)
    F_left = Q_left' ⊗ id(storagetype(C), codomain(C)[2] ⊗ codomain(C)[3])
    F_right = id(storagetype(C), domain(C)[1] ⊗ domain(C)[2]) ⊗ Q_right
    A, B = cluster_complete_svd(F_left' * C * F_right, tol)
    return permute(F_left * A, ((1, 2), (3, 4))),
        permute(B * F_right', ((1, 2), (3, 4)))
end

function cluster_split(center, tol)
    A, B = cluster_complete_svd(permute(center, ((1, 2, 4), (3, 5, 6))), tol)
    return permute(A, ((1, 2), (3, 4))), permute(B, ((1, 2), (3, 4)))
end

"""
    cluster_complete_svd(C, tol)

Factor `C` through its shared SVD support and separate left/right complementary
spaces. For exact rank `r_c`, the virtual multiplicity in sector `c` is
`m_c + n_c - r_c`, the minimum that gives both factors full environment support.
For exact null directions, the factors have the form

    A = [U√S  γU⊥  0],    B = [√S Vᴴ; 0; γV⊥ᴴ].

At a finite rank cutoff, retain the remaining block `E = U⊥' C V⊥` in the
complementary channels, so the product still reconstructs `C` up to roundoff.
The scale `γ` balances these channels against the retained singular values.
"""
function cluster_complete_svd(C::TensorMap, tol::Real)
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

function cluster_contract(O, start::Int, ::Val{1}, boundary::Int)
    return removeunit(removeunit(TensorMap(O[start][1, :, :, boundary]), 4), 1)
end

function cluster_contract(O, start::Int, n::Val{N}, boundary::Int) where {N}
    left = removeunit(TensorMap(O[start][1, :, :, :]), 1)
    right = removeunit(TensorMap(O[start + N - 1][:, :, :, boundary]), 4)
    sites = cluster_bulk(O, start + 1, Val(N - 2))
    return cluster_contract(left, sites, right, n)
end

cluster_bulk(O, start, ::Val{0}) = ()
function cluster_bulk(O, start, ::Val{N}) where {N}
    return (TensorMap(O[start]), cluster_bulk(O, start + 1, Val(N - 1))...)
end

"""
    cluster_contract(left, sites, right, ::Val{N})

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
@generated function cluster_contract(
        left::AbstractTensorMap{<:Any, S, 1, 2}, sites::NTuple{K, AbstractTensorMap{<:Any, S, 2, 2}},
        right::AbstractTensorMap{<:Any, S, 2, 1}, ::Val{N}
    ) where {S, N, K}
    out = tensorexpr(:cluster, -(1:N), -((N + 1):(2N)))
    first_site = tensorexpr(:left, -1, (-N - 1, 1))
    last_site = tensorexpr(:right, (N - 1, -N), -2N)
    middle = [tensorexpr(:(sites[$(i - 1)]), (i - 1, -i), (-N - i, i)) for i in 2:(N - 1)]
    return macroexpand(@__MODULE__, :(return @plansor $out := *($first_site, $(middle...), $last_site)))
end

function cluster_environments(O, start, ::Val{N}) where {N}
    nt = (N - 1) ÷ 2
    return cluster_left_environment(O, start, Val(nt)),
        cluster_right_environment(O, start + N - nt, Val(nt))
end

cluster_left_environment(O, start, ::Val{1}) = O[start][1, 1, 1, 2]
function cluster_left_environment(O, start, ::Val{L}) where {L}
    left = cluster_left_environment(O, start, Val(L - 1))
    return cluster_join_left(left, O[start + L - 1][L, 1, 1, L + 1])
end

cluster_right_environment(O, start, ::Val{1}) = O[start][2, 1, 1, 1]
function cluster_right_environment(O, start, ::Val{L}) where {L}
    right = cluster_right_environment(O, start + 1, Val(L - 1))
    return cluster_join_right(O[start][L + 1, 1, 1, L], right)
end

"""
    cluster_join_left(left, site)

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
@generated function cluster_join_left(left::AbstractTensorMap{<:Any, <:Any, K, K}, site::AbstractTensorMap) where {K}
    out = tensorexpr(:next_left, -(1:(K + 1)), -((K + 2):(2K + 2)))
    env = tensorexpr(:left, -(1:K), (Tuple(-((K + 2):(2K)))..., 1))
    op = tensorexpr(:site, (1, -K - 1), (-2K - 1, -2K - 2))
    return macroexpand(@__MODULE__, :(return @plansor $out := $env * $op))
end

"""
    cluster_join_right(site, right)

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
@generated function cluster_join_right(site::AbstractTensorMap, right::AbstractTensorMap{<:Any, <:Any, K, K}) where {K}
    out = tensorexpr(:next_right, -(1:(K + 1)), -((K + 2):(2K + 2)))
    op = tensorexpr(:site, (-1, -2), (-K - 2, 1))
    env = tensorexpr(:right, (1, Tuple(-(3:(K + 1)))...), -((K + 3):(2K + 2)))
    return macroexpand(@__MODULE__, :(return @plansor $out := $op * $env))
end

function cluster_environment_maps(left, right, ::Val{L}) where {L}
    indices = ntuple(identity, Val(2L + 1))
    return permute(left, (indices, (2L + 2,))),
        permute(right, ((1,), indices .+ 1))
end

function cluster_center(O, start::Int, residual::TensorMap, n::Val{N}) where {N}
    left, right = cluster_environments(O, start, n)
    left_map, right_map = cluster_environment_maps(left, right, Val((N - 1) ÷ 2))
    return cluster_center(left_map, right_map, residual, n)
end

"""
    cluster_center(left_map, right_map, residual, ::Val{N})

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
@generated function cluster_center(
        left_map::TensorMap, right_map::TensorMap, residual::TensorMap, ::Val{N}
    ) where {N}
    nt = (N - 1) ÷ 2
    width = N - 2nt
    # Group all left environment legs into the codomain for the left solve.
    left_indices = (1, Tuple(2:(nt + 1))..., Tuple((N + 2):(N + nt + 1))...)
    remaining = (Tuple((nt + 2):(N + 1))..., Tuple((N + nt + 2):(2N + 2))...)
    # After solving, group the right environment legs into the domain.
    middle_indices = (1, Tuple(2:(width + 1))..., Tuple((nt + width + 2):(nt + 2width + 1))...)
    right_indices = (Tuple((width + 2):(nt + width + 1))..., Tuple((nt + 2width + 2):(2nt + 2width + 2))...)
    center_indices = (Tuple(1:(width + 1)), Tuple((width + 2):(2width + 2)))
    return quote
        left_rhs = permute(residual, $((left_indices, remaining)))
        left_solved = left_map \ left_rhs
        right_rhs = permute(left_solved, $((middle_indices, right_indices)))
        solved = right_rhs / right_map
        return permute(solved, $center_indices)
    end
end
