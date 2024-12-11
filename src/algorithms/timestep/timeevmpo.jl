#https://arxiv.org/pdf/1901.05824.pdf

@kwdef struct WII <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

"""
    TaylorCluster(; N=2, extension=true, compression=true)

Algorithm for constructing the Nth order time evolution MPO using the Taylor cluster expansion.
This is based on the paper [arXiv:2302.14181](https://arxiv.org/abs/2302.14181).
"""
@kwdef struct TaylorCluster <: Algorithm
    N::Int = 2
    extension::Bool = true
    compression::Bool = true
end

const WI = TaylorCluster(; N=1, extension=false, compression=false)

function make_time_mpo(H::MPOHamiltonian, dt::Number, alg::TaylorCluster)
    N = alg.N
    τ = -1im * dt

    # Hack to store FiniteMPOhamiltonians in "square" MPO tensors
    if H isa FiniteMPOHamiltonian
        H′ = copy(H)
        H′[1] = similar(H[2])
        H′[end] = similar(H[end - 1])

        for i in nonzero_keys(H[1])
            H′[1][i] = H[1][i]
        end
        for i in nonzero_keys(H[end])
            H′[end][:, 1, 1, end] = H[end][:, 1, 1, 1]
        end
        H′[1][end, 1, 1, end] += add_util_leg(id(space(H[1][end, 1, 1, end], 2)))
        H′[end][1, 1, 1, 1] += add_util_leg(id(space(H[end][1, 1, 1, 1], 2)))
    else
        H′ = H
    end

    # Check if mpo has the same size everywhere. This is assumed in the following.
    @assert allequal(size.(H′)) "make_time_mpo assumes all mpo tensors to have equal size. A fix for this is yet to be implemented"

    # start with H^N
    H_n = H′^N
    V = size(H′[1], 1)
    linds = LinearIndices(ntuple(i -> V, N))
    cinds = CartesianIndices(linds)

    # extension step: Algorithm 3
    # incorporate higher order terms
    # TODO: don't need to fully construct H_next...
    if alg.extension
        H_next = H_n * H′
        linds_next = LinearIndices(ntuple(i -> V, N + 1))
        cinds_next = CartesianIndices(linds_next)
        for (i, slice) in enumerate(parent(H_n))
            for Inext in nonzero_keys(H_next[i])
                aₑ = cinds_next[Inext[1]]
                bₑ = cinds_next[Inext[4]]
                for c in findall(==(1), aₑ.I), d in findall(==(V), bₑ.I)
                    a = TT.deleteat(Tuple(aₑ), c)
                    b = TT.deleteat(Tuple(bₑ), d)
                    if all(>(1), b) && !(all(in((1, V), a)) && any(==(V), a))
                        n1 = count(==(1), a) + 1
                        n3 = count(==(V), b) + 1
                        factor = τ * factorial(N) / (factorial(N + 1) * n1 * n3)
                        slice[linds[a...], 1, 1, linds[b...]] += factor * H_next[i][Inext]
                    end
                end
            end
        end
    end

    # loopback step: Algorithm 1
    # constructing the Nth order time evolution MPO
    if H isa FiniteMPOHamiltonian
        mpo = FiniteMPO(parent(H_n))
    else
        mpo = InfiniteMPO(parent(H_n))
    end
    for slice in parent(mpo)
        for (I, v) in nonzero_pairs(slice)
            a = cinds[I[1]]
            b = cinds[I[4]]
            if all(in((1, V)), a.I) && any(!=(1), a.I)
                delete!(slice, I)
            elseif any(!=(1), b.I) && all(in((1, V)), b.I)
                exponent = count(==(V), b.I)
                factor = τ^exponent * factorial(N - exponent) / factorial(N)
                Idst = CartesianIndex(Base.setindex(I.I, 1, 4))
                slice[Idst] += factor * v
                delete!(slice, I)
            end
        end
    end

    # Remove equivalent rows and columns: Algorithm 2
    for slice in parent(mpo)
        for I in nonzero_keys(slice)
            a = cinds[I[1]]
            a_sort = CartesianIndex(sort(collect(a.I); by=(!=(1)))...)
            n1_a = count(==(1), a.I)
            n3_a = count(==(V), a.I)
            if n3_a <= n1_a && a_sort != a
                Idst = CartesianIndex(Base.setindex(I.I, linds[a_sort], 1))
                slice[Idst] += slice[I]
                delete!(slice, I)
            elseif a != a_sort
                delete!(slice, I)
            end

            b = cinds[I[4]]
            b_sort = CartesianIndex(sort(collect(b.I); by=(!=(V)))...)
            n1_b = count(==(1), b.I)
            n3_b = count(==(V), b.I)
            if n3_b > n1_b && b_sort != b
                Idst = CartesianIndex(Base.setindex(I.I, linds[b_sort], 4))
                slice[Idst] += slice[I]
                delete!(slice, I)
            elseif b != b_sort
                delete!(slice, I)
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
                for I in nonzero_keys(slice)
                    if I[4] == a_lin
                        Idst = CartesianIndex(Base.setindex(I.I, b_lin, 4))
                        slice[Idst] += factor * slice[I]
                        delete!(slice, I)
                    elseif I[1] == a_lin
                        delete!(slice, I)
                    end
                end
            end
        end
    end

    # Impose boundary conditions as in arXiv:2302.14181
    if mpo isa FiniteMPO
        mpo[1] = mpo[1][1, :, :, :]
        mpo[end] = mpo[end][:, :, :, 1]
    end

    return remove_orphans!(mpo)
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

# function make_time_mpo(H::MPOHamiltonian{S,T}, dt, alg::WII) where {S,T}
#     WA = PeriodicArray{T,3}(undef, H.period, H.odim - 2, H.odim - 2)
#     WB = PeriodicArray{T,2}(undef, H.period, H.odim - 2)
#     WC = PeriodicArray{T,2}(undef, H.period, H.odim - 2)
#     WD = PeriodicArray{T,1}(undef, H.period)
#
#     δ = dt * (-1im)
#
#     for i in 1:(H.period), j in 2:(H.odim - 1), k in 2:(H.odim - 1)
#         init_1 = isometry(storagetype(H[i][1, H.odim]), codomain(H[i][1, H.odim]),
#                           domain(H[i][1, H.odim]))
#         init = [init_1, zero(H[i][1, k]), zero(H[i][j, H.odim]), zero(H[i][j, k])]
#
#         (y, convhist) = exponentiate(1.0, RecursiveVec(init),
#                                      Arnoldi(; tol=alg.tol, maxiter=alg.maxiter)) do x
#             out = similar(x.vecs)
#
#             @plansor out[1][-1 -2; -3 -4] := δ * x[1][-1 1; -3 -4] *
#                                              H[i][1, H.odim][2 3; 1 4] * τ[-2 4; 2 3]
#
#             @plansor out[2][-1 -2; -3 -4] := δ * x[2][-1 1; -3 -4] *
#                                              H[i][1, H.odim][2 3; 1 4] * τ[-2 4; 2 3]
#             @plansor out[2][-1 -2; -3 -4] += sqrt(δ) * x[1][1 2; -3 4] *
#                                              H[i][1, k][-1 -2; 3 -4] * τ[3 4; 1 2]
#
#             @plansor out[3][-1 -2; -3 -4] := δ * x[3][-1 1; -3 -4] *
#                                              H[i][1, H.odim][2 3; 1 4] * τ[-2 4; 2 3]
#             @plansor out[3][-1 -2; -3 -4] += sqrt(δ) * x[1][1 2; -3 4] *
#                                              H[i][j, H.odim][-1 -2; 3 -4] * τ[3 4; 1 2]
#
#             @plansor out[4][-1 -2; -3 -4] := δ * x[4][-1 1; -3 -4] *
#                                              H[i][1, H.odim][2 3; 1 4] * τ[-2 4; 2 3]
#             @plansor out[4][-1 -2; -3 -4] += x[1][1 2; -3 4] * H[i][j, k][-1 -2; 3 -4] *
#                                              τ[3 4; 1 2]
#             @plansor out[4][-1 -2; -3 -4] += sqrt(δ) * x[2][1 2; -3 -4] *
#                                              H[i][j, H.odim][-1 -2; 3 4] * τ[3 4; 1 2]
#             @plansor out[4][-1 -2; -3 -4] += sqrt(δ) * x[3][-1 4; -3 3] *
#                                              H[i][1, k][2 -2; 1 -4] * τ[3 4; 1 2]
#
#             return RecursiveVec(out)
#         end
#         convhist.converged == 0 &&
#             @warn "exponentiate failed to converge: normres = $(convhist.normres)"
#
#         WA[i, j - 1, k - 1] = y[4]
#         WB[i, j - 1] = y[3]
#         WC[i, k - 1] = y[2]
#         WD[i] = y[1]
#     end
#
#     W2 = PeriodicArray{Union{T,Missing},3}(missing, H.period, H.odim - 1, H.odim - 1)
#     W2[:, 2:end, 2:end] = WA
#     W2[:, 2:end, 1] = WB
#     W2[:, 1, 2:end] = WC
#     W2[:, 1, 1] = WD
#
#     return SparseMPO(W2)
# end

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
