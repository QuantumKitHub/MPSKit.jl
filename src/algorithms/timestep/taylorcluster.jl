"""
$(TYPEDEF)

Algorithm for constructing the `N`th order time evolution MPO using the Taylor cluster expansion.

## Fields

$(TYPEDFIELDS)

## References

* [Van Damme et al. SciPost Phys. 17 (2024)](@cite vandamme2024)
"""
@kwdef struct TaylorCluster <: Algorithm
    "order of the Taylor expansion"
    N::Int = 2
    "include higher-order corrections"
    extension::Bool = true
    "approximate compression of corrections, accurate up to order `N`"
    compression::Bool = false
end

"""
    const WI = TaylorCluster(; N=1, extension=false, compression=false)

First order Taylor expansion for a time-evolution MPO.
"""
const WI = TaylorCluster(; N = 1, extension = false, compression = false)

# For type stability reasons, we add a function barrier here and dispatch on Val(N).
# `LinearIndices` and `CartesianIndex{N}` are otherwise abstract.
# However, to not incur a dynamic dispatch for typical cases (N ≤ 4), we manually union-split

function make_time_mpo(
        H::MPOHamiltonian, dt::Number, alg::TaylorCluster;
        tol = eps(real(scalartype(H))), imaginary_evolution::Bool = false
    )
    n = alg.N
    return Base.Cartesian.@nif(
        4,
        d -> n == d,
        d -> _make_time_mpo(H, dt, Val(d), alg.extension, alg.compression; tol, imaginary_evolution),
        d -> _make_time_mpo(H, dt, Val(n), alg.extension, alg.compression; tol, imaginary_evolution),
    )
end

function _make_time_mpo(
        H::MPOHamiltonian, dt::Number, ::Val{N},
        extension::Bool, compression::Bool;
        tol, imaginary_evolution::Bool
    ) where {N}
    τ = imaginary_evolution ? -dt : -1im * dt

    H_n, virtual_sz, linds = _taylor_setup(H, Val(N))

    extension && _taylor_extension!(H_n, H, virtual_sz, linds, Val(N), τ)

    mpo = MPO(map(SparseBlockTensorMap, parent(H_n)))
    _taylor_loopback!(mpo, virtual_sz, linds, Val(N), τ)
    _taylor_remove_equivalents!(mpo, virtual_sz, linds)

    compression && _taylor_compression!(mpo, virtual_sz, linds, Val(N), τ)

    return remove_orphans!(mpo; tol)
end

# `H^N` unrolled with `Val(N)`.
# Mirrors `Base.power_by_squaring`'s tree (and copy-on-N=1 behaviour) so the bond dimensions match `H^N` exactly.
@inline _pow(H, ::Val{1}) = copy(H)
@inline function _pow(H, ::Val{N}) where {N}
    half = _pow(H, Val(N ÷ 2))
    return iseven(N) ? half * half : half * half * H
end

# Stable partition: elements equal to `sentinel` first, others after, preserving relative order within each group.
@inline function _partition_first(t::NTuple{M, Int}, sentinel::Int) where {M}
    n_match = count(==(sentinel), t)
    return ntuple(Val(M)) do j
        if j <= n_match
            _kth_match(t, sentinel, j, true)
        else
            _kth_match(t, sentinel, j - n_match, false)
        end
    end
end

@inline function _kth_match(t::NTuple{M, Int}, sentinel::Int, k::Int, want_match::Bool) where {M}
    cnt = 0
    for x in t
        if (x == sentinel) == want_match
            cnt += 1
            cnt == k && return x
        end
    end
    return 0
end

function _taylor_setup(H::MPOHamiltonian, ::Val{N}) where {N}
    H_n = _pow(H, Val(N))
    virtual_sz = map(0:length(H)) do i
        return i == 0 ? size(H[1], 1) : size(H[i], 4)
    end
    linds = map(virtual_sz) do sz
        return LinearIndices(ntuple(Returns(sz), Val(N)))
    end
    return H_n, virtual_sz, linds
end

# Algorithm 3 of Van Damme et al. (2024): incorporate higher-order corrections
# from `H_next = H_n * H` into `H_n` in place.
function _taylor_extension!(H_n, H, virtual_sz, linds, ::Val{N}, τ::Number) where {N}
    # TODO: don't need to fully construct H_next...
    H_next = H_n * H
    for (i, slice) in enumerate(parent(H_n))
        linds_left = linds[i]
        linds_right = linds[i + 1]
        V_left = virtual_sz[i]
        V_right = virtual_sz[i + 1]
        linds_next_left = LinearIndices(ntuple(Returns(V_left), Val(N + 1)))
        linds_next_right = LinearIndices(ntuple(Returns(V_right), Val(N + 1)))

        for a in CartesianIndices(linds_left), b in CartesianIndices(linds_right)
            all(>(1), b.I) || continue
            all(in((1, V_left)), a.I) && any(==(V_left), a.I) && continue

            n1 = count(==(1), a.I) + 1
            n3 = count(==(V_right), b.I) + 1
            factor = τ * factorial(N) / (factorial(N + 1) * n1 * n3)

            for c in 1:(N + 1), d in 1:(N + 1)
                aₑ = TT.insertafter(a.I, c - 1, (1,))
                bₑ = TT.insertafter(b.I, d - 1, (V_right,))

                # TODO: use VectorInterface for memory efficiency
                slice[linds_left[a], 1, 1, linds_right[b]] += factor *
                    H_next[i][linds_next_left[aₑ...], 1, 1, linds_next_right[bₑ...]]
            end
        end
    end
    return H_n
end

# Algorithm 1: project the auxiliary virtual-bond directions onto the physical
# block, completing the Nth-order time-evolution MPO.
function _taylor_loopback!(mpo, virtual_sz, linds, ::Val{N}, τ::Number) where {N}
    for (i, slice) in enumerate(parent(mpo))
        V_right = virtual_sz[i + 1]
        linds_right = linds[i + 1]
        cinds_right = CartesianIndices(linds_right)
        for b in cinds_right[2:end]
            all(in((1, V_right)), b.I) || continue

            b_lin = linds_right[b]
            a = count(==(V_right), b.I)
            factor = τ^a * factorial(N - a) / factorial(N)
            slice[:, 1, 1, 1] = slice[:, 1, 1, 1] + factor * slice[:, 1, 1, b_lin]
            for I in nonzero_keys(slice)
                (I[1] == b_lin || I[4] == b_lin) && delete!(slice, I)
            end
        end
    end
    return mpo
end

# Algorithm 2: collapse rows and columns that are equivalent under the
# permutation symmetry of the Taylor expansion.
function _taylor_remove_equivalents!(mpo, virtual_sz, linds)
    for (i, slice) in enumerate(parent(mpo))
        V_left = virtual_sz[i]
        linds_left = linds[i]
        for c in CartesianIndices(linds_left)
            c_lin = linds_left[c]
            s_c = CartesianIndex(_partition_first(c.I, 1))

            n1 = count(==(1), c.I)
            n3 = count(==(V_left), c.I)

            if n3 <= n1 && s_c != c
                for k in 1:size(slice, 4)
                    I = CartesianIndex(c_lin, 1, 1, k)
                    if I in nonzero_keys(slice)
                        slice[linds_left[s_c], 1, 1, k] += slice[I]
                        delete!(slice, I)
                    end
                end
            end
        end

        V_right = virtual_sz[i + 1]
        linds_right = linds[i + 1]
        for c in CartesianIndices(linds_right)
            c_lin = linds_right[c]
            s_r = CartesianIndex(_partition_first(c.I, V_right))

            n1 = count(==(1), c.I)
            n3 = count(==(V_right), c.I)

            if n3 > n1 && s_r != c
                for k in 1:size(slice, 1)
                    I = CartesianIndex(k, 1, 1, c_lin)
                    if I in nonzero_keys(slice)
                        slice[k, 1, 1, linds_right[s_r]] += slice[I]
                        delete!(slice, I)
                    end
                end
            end
        end
    end
    return mpo
end

# Algorithm 4: approximate compression — fold the right-going `V_right`
# columns back into the `1`-column with the matching Taylor weight, then drop
# the residual `V_left` rows.
function _taylor_compression!(mpo, virtual_sz, linds, ::Val{N}, τ::Number) where {N}
    for (i, slice) in enumerate(parent(mpo))
        V_right = virtual_sz[i + 1]
        linds_right = linds[i + 1]
        for a in CartesianIndices(linds_right)
            all(>(1), a.I) || continue
            a_lin = linds_right[a]
            n1 = count(==(V_right), a.I)
            n1 == 0 && continue
            b = CartesianIndex(map(x -> x == V_right ? 1 : x, a.I))
            b_lin = linds_right[b]
            factor = τ^n1 * factorial(N - n1) / factorial(N)
            for k in 1:size(slice, 1)
                I = CartesianIndex(k, 1, 1, a_lin)
                if I in nonzero_keys(slice)
                    slice[k, 1, 1, b_lin] += factor * slice[I]
                    delete!(slice, I)
                end
            end
        end
        V_left = virtual_sz[i]
        linds_left = linds[i]
        for a in CartesianIndices(linds_left)
            all(>(1), a.I) || continue
            a_lin = linds_left[a]
            n1 = count(==(V_left), a.I)
            n1 == 0 && continue
            for k in 1:size(slice, 4)
                I = CartesianIndex(a_lin, 1, 1, k)
                delete!(slice, I)
            end
        end
    end
    return mpo
end

# Hack to treat FiniteMPOhamiltonians as Infinite
function make_time_mpo(
        H::FiniteMPOHamiltonian, dt::Number, alg::TaylorCluster;
        tol = eps(real(scalartype(H))), imaginary_evolution::Bool = false
    )
    H′ = copy(parent(H))

    V_left = left_virtualspace(H[1])
    V_left′ = ⊞(V_left, leftunitspace(V_left), leftunitspace(V_left))
    H′[1] = similar(H[1], V_left′ ⊗ space(H[1], 2) ← domain(H[1]))
    for (I, v) in nonzero_pairs(H[1])
        H′[1][I] = v
    end

    V_right = right_virtualspace(H[end])
    V_right′ = ⊞(rightunitspace(V_right), rightunitspace(V_right), V_right)
    H′[end] = similar(H[end], codomain(H[end]) ← space(H[end], 3)' ⊗ V_right′)
    for (I, v) in nonzero_pairs(H[end])
        H′[end][I[1], 1, 1, end] = v
    end

    H′[1][end, 1, 1, end] = H′[1][1, 1, 1, 1]
    H′[end][1, 1, 1, 1] = H′[end][end, 1, 1, end]

    mpo = make_time_mpo(InfiniteMPOHamiltonian(H′), dt, alg; tol, imaginary_evolution)

    # Impose boundary conditions
    mpo_fin = open_boundary_conditions(mpo, length(H))
    return remove_orphans!(mpo_fin; tol)
end
