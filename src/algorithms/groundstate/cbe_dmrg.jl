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

function find_groundstate!(state::AbstractFiniteMPS, O, alg::CBE_DMRG, envs=environments(state, O))
    tol = alg.tol
    maxiter = alg.maxiter
    iter = 0
    delta::Float64 = 2 * tol

    while iter < maxiter && delta > tol
        delta = 0.0
        for l = length(state)-1:-1:1
            Atr = shrewdselect_left(l, state, O, envs, alg.Dprime, alg.Dcirc)
            Aex = state.AL[l] ⊕ Atr
            H = MPO_∂∂AC(
                O[l+1],
                transfer_left(leftenv(envs, l, state), Aex, Aex),
                rightenv(envs, l + 1, state)
            )
            @tensor AC[-1 -2; -3] := conj(Aex[1 2 -1]) * state.AL[l][1 2; 3] * state.AC[l+1][3 -2; -3]
            vals, vecs, info = eigsolve(H, AC, 1, :SR, Lanczos())
            delta = max(delta, calc_galerkin(state, pos, envs))
            state.AC[l] = vecs[1]
        end

        for l = 2:length(state)
            Atr = shrewdselect_right(l, state, O, envs, alg.Dprime, alg.Dcirc)
            Aex = state.AR[l] ⊕ Atr
            H = MPO_∂∂AC(
                O[l-1],
                leftenv(envs, l - 1, state),
                transfer_right(rightenv(envs, l, state), Aex, Aex)
            )
            @tensor AC[-1 -2; -3] := conj(Aex[-3 1 2]) * Aex[3 1 2] * state.AC[l][-1 -2 3]
            vals, vecs, info = eigsolve(H, AC, 1, :SR, Lanczos())
            delta = max(delta, calc_galerkin(state, pos, envs))
            state.AC[l] = vecs[1]
        end
    end
end

# using MUntested
# size(t::AbstractTensorMap, d) = dim(space(t, d))




function shrewdselect_left(l, state, O, envs, Dprime, Dcirc)
    # GR = rightenv(envs, l + 1, state)
    # GR′ = transfer_right(GR, O[l + 1], state.AC[l+1], state.AR[l + 1])
    # Rorth = map(GR′) do gr
    #     @tensor R[-1 -2; -3 -4] := gr[-1 -2 1] * state.AR[l + 1][1 -3 -4]
    # end
    
    
    local Atr
    Rorth = transfer_right_complement(rightenv(envs, l+1, state), O[l + 1], state.AC[l + 1], state.AR[l + 1])
    @floop for R in Rorth
        U, S, _ = tsvd(R, (1,), (2, 3, 4))
        for L in transfer_left_complement(leftenv(envs, l, state), O[l], state.AL[l] * U * S, state.AL[l])
            u′, s′, _ = tsvd(L, (1, 2, 3), (4, ); trunc = truncdim(Dprime))
            û, _, _ = tsvd(u′ * s′, (1, 2), (3, 4), trunc = truncbelow(1e-14))
            û -= state.AL[l] * (state.AL[l]' * û)
            Apr, _, _ = tsvd(û, (1, 2), (3, ), trunc = truncbelow(1e-14))
            
            Ctmp = ∂AC(
                state.AC[l + 1],
                O[l + 1], 
                transfer_left(leftenv(envs, l, state), O[l], state.AL[l], Apr),
                rightenv(envs, l + 1, state)
            )
            @tensor Ctmp[-1 -2; -3] -= Ctmp[-1 1; 2] * conj(state.AR[l + 1][3 1 2]) *
                state.AR[l + 1][3 -2; -3]
            
            ũ, s̃, ṽ = tsvd(Ctmp, (1, ), (2, 3), trunc = truncdim(Dcirc))
            @reduce (Atr += Apr * ũ)
        end
    end
    return Atr
    
#     @tensor begin
#         Rorth[-1 -2; -3 -4] := state.AC[l+1][-1 2; 1] * O[l+1][-2 -3; 2 3] *
#                                rightenv(envs, l + 1, state)[1 3; -4]
#         Rorth[-1 -2; -3 -4] -= Rorth[-1 -2; 1 2] * conj(state.AR[l+1][3 1; 2]) *
#                                state.AR[l+1][3 -3; -4]
#     end

#     U, S, _ = tsvd(Rorth, (1,), (2, 3, 4))
#     ALprime = state

# # size(t::AbstractTensorMap, d) = dim(space(t, d))
#     @tensor begin
#         Lorth[-1 -2; -3 -4] := leftenv(envs, l, state)[-1 3; 1] * ALprime[1 2; -4] *
#                                O[l][2 -3; 2 3]
#         Lorth[-1 -2; -3 -4] -= Lorth[1 2; -3 -4] * conj(state.AL[l][1 2; 3]) *
#                                state.AL[l][-1 -2; 3]
#     end

#     Uprime, Sprime, _ = tsvd(Lorth, (1, 2, 3), (4,); trunc=truncdim(Dprime))

#     Uhat, _, _ = tsvd(Uprime * Sprime, (1, 2), (3, 4), trunc=truncbelow(1e-14))
#     Uhat -= state.AL[l] * (state.AL[l]' * Uhat)

#     Apr, _, _ = tsvd(Uhat, (1, 2), (3,), trunc=truncbelow(1e-14))

#     @tensor begin
#         Lpr[-1 -2; -3] := state.AL[l][1 2; -3] * leftenv(envs, l, state)[4 3 1] * O[l][3 5; 2 -2] * conj(Apr[4 5 -1])
#         Corth[-1; -2 -3] := Lpr[-1 2 1] * state.AC[l+1][1 3; 4] * O[l+1][2 -2; 3 5] *
#                             rightenv(envs, l + 2, state)[4 5; -3]
#         Corth[-1; -2 -3] -= Corth[-1; 1 2] * conj(state.AR[l+1][3 1 2]) * state.AR[l+1][3 -2 -3]
#     end
#     Ucirc, _, _ = tsvd(Corth, (1,), (2, 3), trunc=truncdim(Dcirc))
#     return Apr * Ucirc
end

function shrewdselect_right(l, state, O::DenseMPO, envs, Dprime, Dcirc)
    
end


# function shrewdselect_left(l, state::FiniteMPS{A, B}, O::SparseMPO, envs) where {A,B}
#     GR = rightenv(envs, l + 2, state)
    
#     Rorth = similar(GR, B, length(GR))
#     for j in 1:length(GR)
#         tmp = foldxt(+, 
#             1:length(GR) |>
#             Filter(k -> contains(O[l + 1], j, k)) |>
#             Map() do k
#                 if isscal(O[l + 1], j, k)
#                     return O[l+1].Os[j, k] *
#                         transfer_right_complement(GR[k], state.AC[l + 2], state.AR[l + 2])
#                 else
#                     return transfer_right_complement(GR[k], O[l + 1][j, k],
#                         state.AC[l + 2], state.AR[l + 2])
#                 end
#             end;
#             init = Init(+)
#         )
#         if tmp == Init(+)
#             Rorth[j] = transfer_right_complement(GR[1], H[j, 1], state.AC[l + 2], state.AR[l + 2])
#         else
#             Rorth[j] = tmp
#         end
#         U, S, _ = tsvd(Rorth[j], (1, ), (2, 3, 4))
#         Rorth[j] = U * S
#     end
    
#     GL = leftenv(envs, l, state)
#     Apr = similar(GL, A, length(GL), length(GR))
#     for k in 1:length(GL), n in 1:length(Rorth)
#         tmp = foldxt(+,
#             1:length(GL) |>
#             Filter(j -> contains(O[l], j, k)) |>
#             Map() do j
#                 if isscal(O[l], j, k)
#                     return O[l].Os[j, k] * transfer_left_complement(GL[j], state.AL[l] * Rorth[j], state.AL[l])
#                 else
#                     return transfer_left_complement(GL[j], O[l + 1][j, k],
#                         state.AL[l] * Rorth[j], state.AL[l])
#                 end
#             end;
#             init = Init(+)
#         )
#         if tmp == Init(+)
#             Apr[k,n] = transfer_left_complement(GL[1], H[1, k], state.AL[l] * Rorth[j], state.AL[l])
#         else
#             Apr[k,n] = tmp
#         end
        
#         u′, s′, _ = tsvd(Apr[k,n], (1, 2, 3), (4,), trunc=truncdim(D′))
#         Û, _, _ = tsvd(u′ * s′, (1, 2), (3, 4))
#         Apr[k,n] = Û - state.AL[l] * (state.AL[l]' * Û)
#     end
    
    
#     Atr = foldxt(+, 
#         Map(Apr) do a
#             map(
#                 transfer_left(GL, O[l], state.AL[l], a), 
#                 transfer_right(GR, O[l+1], state.AC[l+1], state.AR[l+1])
#             ) do L, R
#                 utilde, stilde, _ = tsvd(
#                 )
#         end
            
#         end
# end

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

function transfer_left_complement(
    vec::AbstractVector{V}, H::SparseMPOSlice, A::MPSTensor{S}, Ab::MPSTensor{S}
) where {S, V<:MPSTensor{S}}
    return transfer_left_complement(MPOTensor{S}, vec, H, A, Ab)
end

function transfer_right_complement(
    vec::AbstractVector{V}, H::SparseMPOSlice, A::MPSTensor{S}, Ab::MPSTensor{S}
) where {S, V<:MPSTensor{S}}
    return transfer_right_complement(MPOTensor{S}, vec, H, A, Ab)
end

function transfer_left_complement(Lorth_type, vec, H::SparseMPOSlice, A::MPSTensor, Ab::MPSTensor)
    Lorth = similar(vec, Lorth_type, length(vec))
    for k in 1:length(vec)
        res = foldxt(+,
            1:length(vec) |>
            Filter(j -> contains(H, j, k)) |>
            Map() do j
                if isscal(H, j, k)
                    return H.Os[j, k] * transfer_left_complement(vec[j], A, Ab)
                else
                    return transfer_left_complement(vec[j], H[j, k], A, Ab)
                end
            end;
            init = Init(+)
        )
        if res == Init(+)
            Lorth[k] = transfer_left_complement(vec[1], H[1, k], A, Ab)
        else
            Lorth[k] = res
        end
    end
    return Lorth
end

function transfer_right_complement(Rorth_type, vec, H::SparseMPOSlice, A::MPSTensor, Ab::MPSTensor)
    Rorth = similar(vec, Rorth_type, length(vec))
    for j in 1:length(vec)
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
            init = Init(+)
        )
        if res == Init(+)
            Rorth[j] = transfer_right_complement(vec[1], H[j, 1], A, Ab)
        else
            Rorth[j] = res
        end
    end
    return Rorth
end
