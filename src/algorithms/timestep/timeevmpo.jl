"""
$(TYPEDEF)

Generalization of the Euler approximation of the operator exponential for MPOs.

## Fields

$(TYPEDFIELDS)

## References

* [Zaletel et al. Phys. Rev. B 91 (2015)](@cite zaletel2015)
* [Paeckel et al. Ann. of Phys. 411 (2019)](@cite paeckel2019)
"""
@kwdef struct WII <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol
    "maximal number of iterations"
    maxiter::Int = Defaults.maxiter
end

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

@doc """
    make_time_mpo(H::MPOHamiltonian, dt::Number, alg)

Construct an MPO that approximates ``\\exp(-iHdt)``.
""" make_time_mpo

function make_time_mpo(H::MPOHamiltonian, dt::Number, alg::TaylorCluster;
                       tol=eps(real(scalartype(H)))^(3 / 4))
    N = alg.N
    τ = -1im * dt

    # start with H^N
    H_n = H^N
    V = size(H[1], 1)
    linds = LinearIndices(ntuple(i -> V, N))
    cinds = CartesianIndices(linds)

    # extension step: Algorithm 3
    # incorporate higher order terms
    # TODO: don't need to fully construct H_next...
    if alg.extension
        H_next = H_n * H
        linds_next = LinearIndices(ntuple(i -> V, N + 1))
        for (i, slice) in enumerate(parent(H_n))
            for a in cinds, b in cinds
                all(>(1), b.I) || continue
                all(in((1, V)), a.I) && any(==(V), a.I) && continue

                n1 = count(==(1), a.I) + 1
                n3 = count(==(V), b.I) + 1
                factor = τ * factorial(N) / (factorial(N + 1) * n1 * n3)

                for c in 1:(N + 1), d in 1:(N + 1)
                    aₑ = insert!([a.I...], c, 1)
                    bₑ = insert!([b.I...], d, V)

                    # TODO: use VectorInterface for memory efficiency
                    slice[linds[a], 1, 1, linds[b]] += factor *
                                                       H_next[i][linds_next[aₑ...], 1, 1,
                                                                 linds_next[bₑ...]]
                end
            end
        end
    end

    # loopback step: Algorithm 1
    # constructing the Nth order time evolution MPO
    mpo = MPO(parent(H_n))
    for slice in parent(mpo)
        for b in cinds[2:end]
            all(in((1, V)), b.I) || continue

            b_lin = linds[b]
            a = count(==(V), b.I)
            factor = τ^a * factorial(N - a) / factorial(N)
            slice[:, 1, 1, 1] = slice[:, 1, 1, 1] + factor * slice[:, 1, 1, b_lin]
            for I in nonzero_keys(slice)
                (I[1] == b_lin || I[4] == b_lin) && delete!(slice, I)
            end
        end
    end

    # Remove equivalent rows and columns: Algorithm 2
    for slice in parent(mpo)
        for c in cinds
            c_lin = linds[c]
            s_c = CartesianIndex(sort(collect(c.I); by=(!=(1)))...)
            s_r = CartesianIndex(sort(collect(c.I); by=(!=(V)))...)

            n1 = count(==(1), c.I)
            n3 = count(==(V), c.I)

            if n3 <= n1 && s_c != c || n3 > n1 && s_r != c
                if n3 <= n1 && s_c != c
                    for k in 1:size(slice, 4)
                        if CartesianIndex(c_lin, 1, 1, k) in nonzero_keys(slice)
                            slice[linds[s_c], 1, 1, k] += slice[c_lin, 1, 1, k]
                        end
                    end
                elseif n3 > n1 && s_r != c
                    for k in 1:size(slice, 1)
                        if CartesianIndex(k, 1, 1, c_lin) in nonzero_keys(slice)
                            slice[k, 1, 1, linds[s_r]] += slice[k, 1, 1, c_lin]
                        end
                    end
                end
                for I in nonzero_keys(slice)
                    (I[1] == c_lin || I[4] == c_lin) && delete!(slice, I)
                end
            end
        end
    end

    # Approximate compression step: Algorithm 4
    if alg.compression
        for slice in parent(mpo)
            for a in cinds
                all(>(1), a.I) || continue
                a_lin = linds[a]
                n1 = count(==(V), a.I)
                b = CartesianIndex(replace(a.I, V => 1))
                b_lin = linds[b]
                factor = τ^n1 * factorial(N - n1) / factorial(N)
                for k in 1:size(slice, 1)
                    if CartesianIndex(k, 1, 1, a_lin) in nonzero_keys(slice)
                        slice[k, 1, 1, b_lin] += factor * slice[k, 1, 1, a_lin]
                    end
                end

                for I in nonzero_keys(slice)
                    (I[1] == a_lin || I[4] == a_lin) && delete!(slice, I)
                end
            end
        end
    end

    # Impose boundary conditions as in arXiv:2302.14181
    if mpo isa FiniteMPO
        mpo[1] = mpo[1][1, :, :, :]
        mpo[end] = mpo[end][:, :, :, 1]
    end

    return remove_orphans!(mpo; tol=tol)
end

has_prod_elem(slice, t1, t2) = all(map(x -> contains(slice, x...), zip(t1, t2)))
function calc_prod_elem(slice, t1, t2)
    return calc_prod_elem(slice[first(t1), first(t2)], slice, t1[2:end], t2[2:end])
end
function calc_prod_elem(o, slice, t1, t2)
    isempty(t1) && return o

    nel = slice[first(t1), first(t2)]
    fuse_front = isomorphism(fuse(_firstspace(o) * _firstspace(nel)),
                             _firstspace(o) * _firstspace(nel))
    fuse_back = isomorphism(fuse(_lastspace(o)' * _lastspace(nel)'),
                            _lastspace(o)' * _lastspace(nel)')

    @plansor o[-1 -2; -3 -4] := fuse_front[-1; 1 2] * o[1 3; -3 4] * nel[2 -2; 3 5] *
                                conj(fuse_back[-4; 4 5])

    return calc_prod_elem(o, slice, t1[2:end], t2[2:end])
end

function interweave(a, b)
    map(filter(x -> sum(x .== 1) == length(a) && sum(x .== 2) == length(b),
               collect(Iterators.product(fill((1, 2), length(a) + length(b))...)))) do key
        ia = 1
        ib = 1

        output = Vector{eltype(a)}(undef, length(a) + length(b))
        for k in key
            if k == 1
                el = a[ia]
                ia += 1
            else
                el = b[ib]
                ib += 1
            end
            output[ia + ib - 2] = el
        end
        return output
    end
end

function make_time_mpo(H::InfiniteMPOHamiltonian{T}, dt, alg::WII) where {T}
    WA = H.A
    WB = H.B
    WC = H.C
    WD = H.D

    δ = dt * (-1im)
    Wnew = map(1:length(H)) do i
        for j in 2:(size(H[i], 1) - 1), k in 2:(size(H[i], 4) - 1)
            init_1 = isometry(storagetype(WD[i]), codomain(WD[i]), domain(WD[i]))
            init = [init_1,
                    zero(H[i][1, 1, 1, k]),
                    zero(H[i][j, 1, 1, end]),
                    zero(H[i][j, 1, 1, k])]

            y, convhist = exponentiate(1.0, init,
                                       Arnoldi(; tol=alg.tol, maxiter=alg.maxiter)) do x
                out = similar(x)

                @plansor out[1][-1 -2; -3 -4] := δ * x[1][-1 1; -3 -4] *
                                                 H[i][1, 1, 1, end][2 3; 1 4] *
                                                 τ[-2 4; 2 3]

                @plansor out[2][-1 -2; -3 -4] := δ * x[2][-1 1; -3 -4] *
                                                 H[i][1, 1, 1, end][2 3; 1 4] *
                                                 τ[-2 4; 2 3]
                @plansor out[2][-1 -2; -3 -4] += sqrt(δ) *
                                                 x[1][1 2; -3 4] *
                                                 H[i][1, 1, 1, k][-1 -2; 3 -4] *
                                                 τ[3 4; 1 2]

                @plansor out[3][-1 -2; -3 -4] := δ * x[3][-1 1; -3 -4] *
                                                 H[i][1, 1, 1, end][2 3; 1 4] *
                                                 τ[-2 4; 2 3]
                @plansor out[3][-1 -2; -3 -4] += sqrt(δ) *
                                                 x[1][1 2; -3 4] *
                                                 H[i][j, 1, 1, end][-1 -2; 3 -4] *
                                                 τ[3 4; 1 2]

                @plansor out[4][-1 -2; -3 -4] := δ * x[4][-1 1; -3 -4] *
                                                 H[i][1, 1, 1, end][2 3; 1 4] *
                                                 τ[-2 4; 2 3]
                @plansor out[4][-1 -2; -3 -4] += x[1][1 2; -3 4] *
                                                 H[i][j, 1, 1, k][-1 -2; 3 -4] *
                                                 τ[3 4; 1 2]
                @plansor out[4][-1 -2; -3 -4] += sqrt(δ) *
                                                 x[2][1 2; -3 -4] *
                                                 H[i][j, 1, 1, end][-1 -2; 3 4] *
                                                 τ[3 4; 1 2]
                @plansor out[4][-1 -2; -3 -4] += sqrt(δ) *
                                                 x[3][-1 4; -3 3] *
                                                 H[i][1, 1, 1, k][2 -2; 1 -4] *
                                                 τ[3 4; 1 2]

                return out
            end
            convhist.converged == 0 &&
                @warn "failed to exponentiate $(convhist.normres)"

            WA[i][j - 1, 1, 1, k - 1] = y[4]
            WB[i][j - 1, 1, 1, 1] = y[3]
            WC[i][1, 1, 1, k - 1] = y[2]
            WD[i] = y[1]
        end

        Vₗ = left_virtualspace(H, i)[1:(end - 1)]
        Vᵣ = right_virtualspace(H, i)[1:(end - 1)]
        P = physicalspace(H, i)

        h′ = similar(H[i], Vₗ ⊗ P ← P ⊗ Vᵣ)
        h′[2:end, 1, 1, 2:end] = WA[i]
        h′[2:end, 1, 1, 1] = WB[i]
        h′[1, 1, 1, 2:end] = WC[i]
        h′[1, 1, 1, 1] = WD[i]

        return h′
    end

    return InfiniteMPO(PeriodicArray(Wnew))
end
