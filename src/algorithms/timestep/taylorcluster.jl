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
const WI = TaylorCluster(; N=1, extension=false, compression=false)

function make_time_mpo(H::MPOHamiltonian, dt::Number, alg::TaylorCluster;
                       tol=eps(real(scalartype(H))))
    N = alg.N
    τ = -1im * dt

    # start with H^N
    H_n = H^N
    virtual_sz = map(0:length(H)) do i
        return i == 0 ? size(H[1], 1) : size(H[i], 4)
    end
    linds = map(virtual_sz) do sz
        return LinearIndices(ntuple(Returns(sz), N))
    end

    # extension step: Algorithm 3
    # incorporate higher order terms
    # TODO: don't need to fully construct H_next...
    if alg.extension
        H_next = H_n * H
        for (i, slice) in enumerate(parent(H_n))
            linds_left = linds[i]
            linds_right = linds[i + 1]
            V_left = virtual_sz[i]
            V_right = virtual_sz[i + 1]
            linds_next_left = LinearIndices(ntuple(Returns(V_left), N + 1))
            linds_next_right = LinearIndices(ntuple(Returns(V_right), N + 1))

            for a in CartesianIndices(linds_left), b in CartesianIndices(linds_right)
                all(>(1), b.I) || continue
                all(in((1, V_left)), a.I) && any(==(V_left), a.I) && continue

                n1 = count(==(1), a.I) + 1
                n3 = count(==(V_right), b.I) + 1
                factor = τ * factorial(N) / (factorial(N + 1) * n1 * n3)

                for c in 1:(N + 1), d in 1:(N + 1)
                    aₑ = insert!([a.I...], c, 1)
                    bₑ = insert!([b.I...], d, V_right)

                    # TODO: use VectorInterface for memory efficiency
                    slice[linds_left[a], 1, 1, linds_right[b]] += factor *
                                                                  H_next[i][linds_next_left[aₑ...],
                                                                            1, 1,
                                                                            linds_next_right[bₑ...]]
                end
            end
        end
    end

    # loopback step: Algorithm 1
    # constructing the Nth order time evolution MPO
    mpo = MPO(parent(H_n))
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

    # Remove equivalent rows and columns: Algorithm 2
    for (i, slice) in enumerate(parent(mpo))
        V_left = virtual_sz[i]
        linds_left = linds[i]
        for c in CartesianIndices(linds_left)
            c_lin = linds_left[c]
            s_c = CartesianIndex(sort(collect(c.I); by=(!=(1)))...)

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
            s_r = CartesianIndex(sort(collect(c.I); by=(!=(V_right)))...)

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

    # Approximate compression step: Algorithm 4
    if alg.compression
        for (i, slice) in enumerate(parent(mpo))
            V_right = virtual_sz[i + 1]
            linds_right = linds[i + 1]
            for a in CartesianIndices(linds_right)
                all(>(1), a.I) || continue
                a_lin = linds_right[a]
                n1 = count(==(V_right), a.I)
                n1 == 0 && continue
                b = CartesianIndex(replace(a.I, V_right => 1))
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
    end

    return remove_orphans!(mpo; tol)
end

# Hack to treat FiniteMPOhamiltonians as Infinite
function make_time_mpo(H::FiniteMPOHamiltonian, dt::Number, alg::TaylorCluster;
                       tol=eps(real(scalartype(H))))
    H′ = copy(parent(H))

    V_left = left_virtualspace(H[1])
    V_left′ = BlockTensorKit.oplus(V_left, oneunit(V_left), oneunit(V_left))
    H′[1] = similar(H[1], V_left′ ⊗ space(H[1], 2) ← domain(H[1]))
    for (I, v) in nonzero_pairs(H[1])
        H′[1][I] = v
    end

    V_right = right_virtualspace(H[end])
    V_right′ = BlockTensorKit.oplus(oneunit(V_right), oneunit(V_right), V_right)
    H′[end] = similar(H[end], codomain(H[end]) ← space(H[end], 3)' ⊗ V_right′)
    for (I, v) in nonzero_pairs(H[end])
        H′[end][I[1], 1, 1, end] = v
    end

    H′[1][end, 1, 1, end] = H′[1][1, 1, 1, 1]
    H′[end][1, 1, 1, 1] = H′[end][end, 1, 1, end]

    mpo = make_time_mpo(InfiniteMPOHamiltonian(H′), dt, alg; tol)

    # Impose boundary conditions
    mpo_fin = open_boundary_conditions(mpo, length(H))
    return remove_orphans!(mpo_fin; tol)
end
