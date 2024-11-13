#https://arxiv.org/pdf/1901.05824.pdf

@kwdef struct WII <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

struct TaylorCluster{N} <: Algorithm end

const WI = TaylorCluster{1};

function make_time_mpo(th::MPOHamiltonian{S,T,E}, dt, alg::TaylorCluster{N}) where {S,T,E,N}
    τ = -1im * dt

    inds = LinearIndices(ntuple(i -> th.odim, N))
    mult_data = Array{Union{Missing,eltype(th[1])},3}(missing, length(th), th.odim^N,
                                                      th.odim^N)
    for loc in 1:length(th), a in CartesianIndices(inds), b in CartesianIndices(inds)
        has_prod_elem(th[loc], Tuple(a), Tuple(b)) || continue

        mult_data[loc, inds[a], inds[b]] = calc_prod_elem(th[loc], Tuple(a), Tuple(b))
    end
    mult = SparseMPO(mult_data)

    for slice in mult
        #embed next order in this one - incompatible with approximate compression
        for a in CartesianIndices(inds), b in CartesianIndices(inds), no in 1:1
            t_a = [Tuple(a)...]
            t_b = [Tuple(b)...]

            all(x -> x > 1, t_b) || continue
            all(x -> x in (1, th.odim), t_a) && any(x -> x == th.odim, t_a) && continue

            n3 = count(x -> x == th.odim, t_b) + no
            n1 = count(x -> x == 1, t_a) + no
            for e_b in interweave(fill(th.odim, no), t_b),
                e_a in interweave(fill(1, no), t_a)

                has_prod_elem(slice, e_a, e_b) || continue
                slice[inds[a], inds[b]] += calc_prod_elem(slice, e_a, e_b) * τ^no *
                                           factorial(N) / (factorial(N + no) * n1 * n3)
            end
        end

        # apply loopback
        for a in Iterators.product(fill((1, th.odim), N)...)
            all(a .== 1) && continue

            order = count(x -> x == th.odim, a)
            c_ind = inds[a...]
            slice[1:(c_ind - 1), 1] .+= slice[1:(c_ind - 1), c_ind] .* τ^order *
                                        factorial(N - order) / factorial(N)
            slice[c_ind, :] .*= 0
            slice[:, c_ind] .*= 0
        end

        # remove equivalent collumns
        for c in CartesianIndices(inds)
            tc = [Tuple(c)...]
            keys = map(x -> x == 1 ? 2 : 1, tc)
            s_tc = tc[sortperm(keys)]

            n1 = count(x -> x == 1, tc)
            n3 = count(x -> x == th.odim, tc)

            if n1 >= n3 && tc != s_tc
                slice[inds[s_tc...], :] += slice[inds[c], :]

                slice[inds[c], :] .*= 0
                slice[:, inds[c]] .*= 0
            end
        end

        # remove equivalent rows
        for c in CartesianIndices(inds)
            tc = [Tuple(c)...]
            keys = map(x -> x == th.odim ? 2 : 1, tc)
            s_tc = tc[sortperm(keys)]

            n1 = count(x -> x == 1, tc)
            n3 = count(x -> x == th.odim, tc)

            if n3 > n1 && tc != s_tc
                slice[:, inds[s_tc...]] += slice[:, inds[c]]

                slice[:, inds[c]] .*= 0
                slice[inds[c], :] .*= 0
            end
        end
        # approximate compression
        for c in CartesianIndices(inds)
            tc = [Tuple(c)...]

            n = count(x -> x == th.odim, tc)
            all(x -> x > 1, tc) && n > 0 || continue

            transformed = map(x -> x == th.odim ? 1 : x, tc)

            slice[:, inds[transformed...]] += slice[:, inds[tc...]] * τ^n *
                                              factorial(N - n) / factorial(N)

            slice[:, inds[tc...]] .*= 0
            slice[inds[tc...], :] .*= 0
        end
    end

    return remove_orphans(mult)
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

function make_time_mpo(H::MPOHamiltonian{S,T}, dt, alg::WII) where {S,T}
    WA = PeriodicArray{T,3}(undef, H.period, H.odim - 2, H.odim - 2)
    WB = PeriodicArray{T,2}(undef, H.period, H.odim - 2)
    WC = PeriodicArray{T,2}(undef, H.period, H.odim - 2)
    WD = PeriodicArray{T,1}(undef, H.period)

    δ = dt * (-1im)

    for i in 1:(H.period), j in 2:(H.odim - 1), k in 2:(H.odim - 1)
        init_1 = isometry(storagetype(H[i][1, H.odim]), codomain(H[i][1, H.odim]),
                          domain(H[i][1, H.odim]))
        init = [init_1, zero(H[i][1, k]), zero(H[i][j, H.odim]), zero(H[i][j, k])]

        (y, convhist) = exponentiate(1.0, init,
                                     Arnoldi(; tol=alg.tol, maxiter=alg.maxiter)) do x
            out = similar(x)

            @plansor out[1][-1 -2; -3 -4] := δ * x[1][-1 1; -3 -4] *
                                             H[i][1, H.odim][2 3; 1 4] * τ[-2 4; 2 3]

            @plansor out[2][-1 -2; -3 -4] := δ * x[2][-1 1; -3 -4] *
                                             H[i][1, H.odim][2 3; 1 4] * τ[-2 4; 2 3]
            @plansor out[2][-1 -2; -3 -4] += sqrt(δ) * x[1][1 2; -3 4] *
                                             H[i][1, k][-1 -2; 3 -4] * τ[3 4; 1 2]

            @plansor out[3][-1 -2; -3 -4] := δ * x[3][-1 1; -3 -4] *
                                             H[i][1, H.odim][2 3; 1 4] * τ[-2 4; 2 3]
            @plansor out[3][-1 -2; -3 -4] += sqrt(δ) * x[1][1 2; -3 4] *
                                             H[i][j, H.odim][-1 -2; 3 -4] * τ[3 4; 1 2]

            @plansor out[4][-1 -2; -3 -4] := δ * x[4][-1 1; -3 -4] *
                                             H[i][1, H.odim][2 3; 1 4] * τ[-2 4; 2 3]
            @plansor out[4][-1 -2; -3 -4] += x[1][1 2; -3 4] * H[i][j, k][-1 -2; 3 -4] *
                                             τ[3 4; 1 2]
            @plansor out[4][-1 -2; -3 -4] += sqrt(δ) * x[2][1 2; -3 -4] *
                                             H[i][j, H.odim][-1 -2; 3 4] * τ[3 4; 1 2]
            @plansor out[4][-1 -2; -3 -4] += sqrt(δ) * x[3][-1 4; -3 3] *
                                             H[i][1, k][2 -2; 1 -4] * τ[3 4; 1 2]

            return out
        end
        convhist.converged == 0 &&
            @warn "exponentiate failed to converge: normres = $(convhist.normres)"

        WA[i, j - 1, k - 1] = y[4]
        WB[i, j - 1] = y[3]
        WC[i, k - 1] = y[2]
        WD[i] = y[1]
    end

    W2 = PeriodicArray{Union{T,Missing},3}(missing, H.period, H.odim - 1, H.odim - 1)
    W2[:, 2:end, 2:end] = WA
    W2[:, 2:end, 1] = WB
    W2[:, 1, 2:end] = WC
    W2[:, 1, 1] = WD

    return SparseMPO(W2)
end
