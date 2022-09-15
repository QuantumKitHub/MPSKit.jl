"""
    struct CBE_DMRG{F} <: Algorithm

Controlled bond expansion for DMRG ground state search at single-site costs.
(https://arxiv.org/abs/2207.14712)
    
# Fields
- `tol = Defaults.tol`:

"""
@with_kw struct CBE_DMRG{F} <: Algorithm
    tol = Defaults.tol
    maxiter = Defaults.maxiter
    Dprime::Int = 20
    Dcirc::Int = 20
    verbose = Defaults.verbose
    finalize::F = Defaults._finalize
end


function find_groundstate(state, H, alg::CBE_DMRG, envs=environments(state, H))
    return find_groundstate!(copy(state), H, alg, envs)
end



using MUntested
size(t::AbstractTensorMap, i) = dim(space(t, i))

function find_groundstate!(state::AbstractFiniteMPS, O, alg::CBE_DMRG, envs=environments(state, O))
    tol = alg.tol
    maxiter = alg.maxiter
    iter = 0
    delta::Float64 = 2 * tol

    while iter < maxiter && delta > tol
        delta = 0.0
        for l = length(state)-1:-1:1
            Aex = shrewdselect_left(l, state, O, envs, alg.Dprime, alg.Dcirc)
            @info "expanded bond $l from $(space(state.AR[l], 3)) to $(space(Aex, 3))"
            H = MPO_∂∂AC(
                O[l+1],
                transfer_left(leftenv(envs, l, state), O[l], Aex, Aex),
                rightenv(envs, l + 1, state)
            )
            @tensor AC[-1 -2; -3] := conj(Aex[1 2 -1]) * state.AL[l][1 2; 3] * state.AC[l+1][3 -2; -3]
            vals, vecs, info = eigsolve(H, AC, 1, :SR, Lanczos())
            delta = max(delta, calc_galerkin(state, l, envs))

            L, Q = rightorth!(_transpose_tail(vecs[1]), alg=LQpos())

            state.AC[l+1] = (L, _transpose_front(Q))
            state.AC[l] = (Aex, L)
        end

        for l = 2:length(state)
            Aex = shrewdselect_right(l, state, O, envs, alg.Dprime, alg.Dcirc)
            @info " expanded bond $l from $(space(state.AR[l], 3)) to $(space(Aex, 3))"
            H = MPO_∂∂AC(
                O[l-1],
                leftenv(envs, l - 1, state),
                transfer_right(rightenv(envs, l, state), O[l], Aex, Aex)
            )
            @otensor AC[-1 -2; -3] := conj(Aex[-3 1; 2]) * state.AR[l][3 1; 2] * state.AC[l-1][-1 -2; 3]
            vals, vecs, info = eigsolve(H, AC, 1, :SR, Lanczos())
            delta = max(delta, calc_galerkin(state, l, envs))

            Q, R = leftorth!(vecs[1], alg=QRpos())
            state.AC[l-1] = (Q, R)
            state.AC[l] = (R, Aex)
        end
        alg.verbose && @info "Iteraton $(iter) error $(delta)"
        flush(stdout)
        #finalize
        (state, envs) = alg.finalize(iter, state, O, envs)::Tuple{typeof(state),typeof(envs)}
        iter += 1
    end
    return state, envs, delta
end



function shrewdselect_left(l, state, O, envs, Dprime, Dcirc)
    Rorth = transfer_right_complement(rightenv(envs, l + 1, state), O[l+1], state.AC[l+1], state.AR[l+1])
    norm(Rorth) < 1e-14 && return state.AL[l]
    U, S, _ = tsvd(Rorth, (1,), (2, 3, 4), trunc=truncbelow(1e-14))
    Lorth = transfer_left_complement(leftenv(envs, l, state), O[l], state.AL[l] * U * S, state.AL[l])
    norm(Lorth) < 1e-14 && return state.AL[l]
    u′, s′, _ = tsvd(Lorth, (1, 2, 3), (4,); trunc=truncdim(Dprime))
    û, _, _ = tsvd(u′ * s′, (1, 2), (3, 4); trunc=truncbelow(1e-14))
    û -= state.AL[l] * (state.AL[l]' * û)
    Apr, _, _ = tsvd(û, (1, 2), (3,); trunc=truncbelow(1e-14))

    Ctmp = ∂AC(
        state.AC[l+1],
        O[l+1],
        transfer_left(leftenv(envs, l, state), O[l], state.AL[l], Apr),
        rightenv(envs, l + 1, state)
    )
    @tensor Ctmp[-1 -2; -3] -= Ctmp[-1 1; 2] * conj(state.AR[l+1][3 1 2]) * state.AR[l+1][3 -2 -3]

    ũ, s̃, ṽ = tsvd(Ctmp, (1,), (2, 3), trunc=truncdim(Dcirc))
    return Aex = catdomain(state.AL[l], (Apr * ũ))
end

function shrewdselect_right(l, state, O, envs, Dprime, Dcirc)
    Lorth = transfer_left_complement(leftenv(envs, l - 1, state), O[l-1], state.AC[l-1], state.AL[l-1])
    norm(Lorth) < 1e-14 && return state.AR[l]
    _, S, V = tsvd(Lorth, (1, 2, 3), (4,); trunc=truncbelow(1e-14))
    @tensor AR′[-1 -2; -3] := S[-1; 1] * V[1; 2] * state.AR[l][2 -2; -3]
    Rorth = transfer_right_complement(rightenv(envs, l, state), O[l], AR′, state.AR[l])
    norm(Rorth) < 1e-14 && return state.AR[l]
    _, s′, v′ = tsvd(Rorth, (1,), (2, 3, 4); trunc=truncdim(Dprime))
    _, _, v̂ = tsvd(s′ * v′, (1, 2), (3, 4); trunc=truncbelow(1e-14))
    @tensor v̂[-1; -2 -3] -= v̂[-1; 1 2] * conj(state.AR[l][3 1; 2]) * state.AR[l][3 -2; -3]
    
    # v̂ -= (v̂ * _transpose_tail(state.AR[l])') * _transpose_tail(state.AR[l])
    Apr, _, _ = tsvd(v̂, (1, 2), (3,); trunc=truncbelow(1e-14))

    Ctmp = ∂AC(
        state.AC[l-1],
        O[l-1],
        leftenv(envs, l - 1, state),
        transfer_right(rightenv(envs, l, state), O[l], state.AR[l], Apr)
    )
    @tensor Ctmp[-1 -2; -3] -= Ctmp[1 2; -3] * conj(state.AL[l-1][1 2; 3]) * state.AL[l-1][-1 -2; 3]

    ũ, s̃, ṽ = tsvd(Ctmp, (1, 2), (3,), trunc=truncdim(Dcirc))
    return Aex = _transpose_front(catcodomain(_transpose_tail(state.AR[l]), ṽ * _transpose_tail(Apr)))
end

transfer_left_complement(v, ::Nothing, A, B) = transfer_left_complement(v, A, B)

transfer_right_complement(v, ::Nothing, A, B) = transfer_right_complement(v, A, B)

function transfer_left_complement(v::MPSTensor, A::MPSTensor, Ab::MPSTensor)
    @tensor begin
        Lorth[-1 -2; -3 -4] := v[-1 -3; 1] * A[1 -2; -4]
        Lorth[-1 -2; -3 -4] -= Lorth[1 2; -3 -4] * conj(Ab[1 2; 3]) * Ab[-1 -2; 3]
    end
    return Lorth
end

function transfer_right_complement(v::MPSTensor, A::MPSTensor, Ab::MPSTensor)
    @tensor begin
        Rorth[-1 -2; -3 -4] := A[-1 -3; 1] * v[1 -2; -4]
        Rorth[-1 -2; -3 -4] -= Rorth[-1 -2; 1 2] * conj(Ab[3 1; 2]) * Ab[3 -3; -4]
    end
    return Rorth
end

function transfer_left_complement(v::MPSTensor, O::MPOTensor, A::MPSTensor, Ab::MPSTensor)
    @tensor begin
        Lorth[-1 -2; -3 -4] := v[-1 2; 1] * A[1 3; -4] * O[2 -2; 3 -3]
        Lorth[-1 -2; -3 -4] -= Lorth[1 2; -3 -4] * conj(Ab[1 2; 3]) * Ab[-1 -2; 3]
    end
    return Lorth
end

function transfer_right_complement(v::MPSTensor, O::MPOTensor, A::MPSTensor, Ab::MPSTensor)
    @tensor begin
        Rorth[-1 -2; -3 -4] := A[-1 2; 1] * O[-2 -3; 2 3] * v[1 3; -4]
        Rorth[-1 -2; -3 -4] -= Rorth[-1 -2; 1 2] * conj(Ab[3 1; 2]) * Ab[3 -3; -4]
    end
    return Rorth
end

function transfer_left_complement(vec, H::SparseMPOSlice, A::MPSTensor, Ab::MPSTensor)
    Lorth = mapfoldl(catdomain, 1:length(vec)) do k
        res = foldxt(+,
            1:length(vec) |> Filter(j -> contains(H, j, k)) |> Map() do j
                if isscal(H, j, k)
                    return H.Os[j, k] * transfer_left_complement(vec[j], A, Ab)
                else
                    return transfer_left_complement(vec[j], H[j, k], A, Ab)
                end
            end;
            init=Init(+)
        )
        if res == Init(+)
            tmp = transfer_left_complement(vec[1], H[1, k], A, Ab)
        else
            tmp = res
        end
        return permute(tmp, (1, 2, 4), (3,))
    end
    return permute(Lorth, (1, 2), (4, 3))
end

function transfer_right_complement(vec, H::SparseMPOSlice, A::MPSTensor, Ab::MPSTensor)
    Rorth = mapfoldl(catdomain, 1:length(vec)) do j
        res = foldxt(+,
            1:length(vec) |>
            Filter(k -> contains(H, j, k)) |>
            Map() do k
                if isscal(H, j, k)
                    return H.Os[j, k] * transfer_right_complement(vec[k], A, Ab)
                else
                    return transfer_right_complement(vec[k], H[j, k], A, Ab)
                end
            end;
            init=Init(+)
        )
        if res == Init(+)
            tmp = transfer_right_complement(vec[1], H[j, 1], A, Ab)
        else
            tmp = res
        end
        return permute(tmp, (1, 3, 4), (2,))
    end
    return permute(Rorth, (1, 4), (2, 3))
end
