abstract type NoiseSchedule end

struct ExponentialDecay <: NoiseSchedule
    decay_rate::Float64
end

struct Warmup <: NoiseSchedule
    iters::Int
end

(s::ExponentialDecay)(noise, iter, ϵ) = noise * s.decay_rate^iter
(s::Warmup)(noise, iter, ϵ) = iter ≤ s.iters ? noise : zero(noise)

struct DMRG3S{N, S <: NoiseSchedule} <: Algorithm
    noise::N
    schedule::S
end

function _update_post_expand(alg::DMRG3S, iter, ϵ)
    noise = alg.schedule(alg.noise, iter, ϵ)
    return iszero(noise) ? NoExpand() : DMRG3S(noise, alg.schedule)
end

function _get_combiner(::Type{T}, V1, V2) where {T}
    Vf = fuse(V1 ⊗ V2)
    return isomorphism(T, (V1 ⊗ V2) ← Vf), Vf
end

function post_expand!(pos::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, alg::DMRG3S, envs, AC, alg_gauge; normalize=true)
    El = leftenv(envs, pos, ψ)
    Hi = H[pos]
    α = alg.noise
    T = promote_type(scalartype(ψ), scalartype(Hi))
    V = right_virtualspace(AC)
    combiner, Vpert = _get_combiner(T, V, right_virtualspace(Hi))
    
    @plansor pert[-1 -2; -3] := α * El[-1 1; 2] * AC[2 3; 4] * Hi[1 -2; 3 5] * combiner[4, 5; -3]    

    AC_expanded = catdomain(AC, pert)

    AL, C = left_gauge(AC_expanded, alg_gauge)
    B = _transpose_tail(ψ.AR[pos+1])
    AR = _transpose_front(catcodomain(B, zeros(T, Vpert ← domain(B))))

    normalize && normalize!(C)
    ψ.AC[pos] = (AL, C)
    ψ.AC[pos + 1] = (C, AR)
    return ψ
end

function post_expand!(pos::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, alg::DMRG3S, envs, AC, alg_gauge; normalize=true)
    Er = rightenv(envs, pos, ψ)
    Hi = H[pos]
    α = alg.noise
    T = promote_type(scalartype(ψ), scalartype(Hi))
    V = left_virtualspace(AC)
    combiner, Vpert = _get_combiner(T, V, left_virtualspace(Hi))
    
    @plansor pert[l; r s] := α * (combiner')[l; li lh] * AC[li, si; ri] * Hi[lh, s; si, rh] * Er[ri rh; r]

    AC = _transpose_tail(AC)
    AC_expanded = catcodomain(AC, pert)

    C, AR = right_gauge(AC_expanded, alg_gauge)
    B = ψ.AL[pos-1]
    AL = catdomain(B, zeros(T, codomain(B) ← Vpert))
    
    normalize && normalize!(C)
    ψ.AC[pos] = (C, AR)
    ψ.AC[pos - 1] = (AL, C)
    return ψ
end