"""
    NoiseSchedule

Supertype for controlling how a bond-expansion algorithm's noise/perturbation amplitude
evolves across DMRG sweeps. A schedule `s` is called as `s(noise, iter, ϵ) -> noise`, given
the current noise amplitude, the outer iteration count, and the current global convergence
error, and returns the amplitude to use for the next iteration. Schedules can be composed
with `∘`; see [`ExponentialDecay`](@ref), [`Warmup`](@ref), [`FunctionalSchedule`](@ref).
"""
abstract type NoiseSchedule end

"""
    FunctionalSchedule(f) <: NoiseSchedule

Wrap an arbitrary callable `f(noise, iter, ϵ) -> noise` as a [`NoiseSchedule`](@ref).
Used internally by `∘` to compose schedules, but can also be constructed directly for
ad hoc schedules that don't warrant their own named type.
"""
struct FunctionalSchedule{F} <: NoiseSchedule
    f::F
end
(s::FunctionalSchedule)(noise, iter, ϵ) = s.f(noise, iter, ϵ)

"""
    s1::NoiseSchedule ∘ s2::NoiseSchedule

Compose two noise schedules: `s2` is applied first, and its result is fed through `s1`,
i.e. `(s1 ∘ s2)(noise, iter, ϵ) == s1(s2(noise, iter, ϵ), iter, ϵ)` — the same convention
as function composition in `Base`.
"""
Base.:∘(s1::NoiseSchedule, s2::NoiseSchedule) =
    FunctionalSchedule((noise, iter, ϵ) -> s1(s2(noise, iter, ϵ), iter, ϵ))

"""
    ExponentialDecay(decay_rate; threshold = 0.0)

Noise schedule that shrinks geometrically: `noise -> noise * decay_rate^iter`, snapped
to exactly zero once it falls below `threshold`. Use `decay_rate < 1` for a standalone
[`DMRG3S`](@ref) run that gradually turns enrichment off as the state converges; a
nonzero `threshold` avoids running the (cheap, but non-free) expansion step
indefinitely on a noise amplitude too small to matter.
"""
struct ExponentialDecay{T <: Real} <: NoiseSchedule
    decay_rate::T
    threshold::T
end
ExponentialDecay(decay_rate, threshold) = ExponentialDecay(promote(decay_rate, threshold)...)
ExponentialDecay(decay_rate; threshold = zero(decay_rate)) = ExponentialDecay(decay_rate, threshold)

function (s::ExponentialDecay)(noise, iter, ϵ)
    decayed = noise * s.decay_rate^iter
    return decayed < s.threshold ? zero(decayed) : decayed
end

"""
    Warmup(iters::Int)

Noise schedule that keeps the noise amplitude constant for the first `iters` outer
iterations, then drops it to exactly zero. Useful composed with a decaying schedule (e.g.
`ExponentialDecay(0.7) ∘ Warmup(5)`) to hold the perturbation at full strength for a few
sweeps before starting to taper it off.
"""
struct Warmup <: NoiseSchedule
    iters::Int
end

(s::Warmup)(noise, iter, ϵ) = iter ≤ s.iters ? noise : zero(noise)

"""
    DMRG3S(noise, schedule::NoiseSchedule)

Gauge algorithm wrapper that, at every site update, injects a Hamiltonian-derived
perturbation of the just-optimized tensor before gauging — the "strictly single-site DMRG with
subspace expansion" scheme of Hubig, McCulloch, Schollwöck & Wolf,
[Phys. Rev. B 91, 155115 (2015)](https://doi.org/10.1103/PhysRevB.91.155115). This lets
single-site DMRG introduce basis states/quantum-number sectors absent from the initial
state, helping it escape local minima that plain single-site DMRG can get stuck in.

`noise` is the initial perturbation amplitude; `schedule` (see [`ExponentialDecay`](@ref),
[`Warmup`](@ref)) controls how it evolves across outer iterations, and once it decays to
exactly zero the gauge step reverts to a plain [`NoExpand`](@ref) for the remainder of the
run. As with [`NoExpand`](@ref), the actual factorization used to gauge the enriched tensor
is filled in by `DMRG`'s constructor, not supplied here directly — see `DMRG`'s docstring
for the calling convention:

```julia
DMRG(; alg_gauge = DMRG3S(0.1, ExponentialDecay(0.7)), trscheme = truncdim(50))
```

A truncating `trscheme` is strongly recommended alongside `DMRG3S`, to cut the perturbed
bond back down each sweep — `DMRG`'s constructor warns if none is given.
"""
struct DMRG3S{N, S <: NoiseSchedule, A} <: Algorithm
    noise::N
    schedule::S
    alg_gauge::A
end

DMRG3S(noise, schedule) = DMRG3S(noise, schedule, nothing)

set_alg_gauge(alg::DMRG3S, inner_gauge) = DMRG3S(alg.noise, alg.schedule, inner_gauge)

function _update_alg_gauge(alg::DMRG3S, iter, ϵ)
    noise = alg.schedule(alg.noise, iter, ϵ)
    return iszero(noise) ? NoExpand(alg.alg_gauge) : DMRG3S(noise, alg.schedule, alg.alg_gauge)
end

function _get_combiner(::Type{T}, V1, V2) where {T}
    Vf = fuse(V1 ⊗ V2)
    return isomorphism(T, (V1 ⊗ V2) ← Vf), Vf
end

function gauge!(pos::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, envs, AC, alg::DMRG3S; normalize = true)
    El = leftenv(envs, pos, ψ)
    Hi = H[pos]
    α = alg.noise
    T = promote_type(scalartype(ψ), scalartype(Hi))
    V = right_virtualspace(AC)
    combiner, Vpert = _get_combiner(T, V, right_virtualspace(Hi))

    @plansor pert[-1 -2; -3] := α * El[-1 1; 2] * AC[2 3; 4] * Hi[1 -2; 3 5] * combiner[4, 5; -3]

    AC_expanded = catdomain(AC, pert)

    AL, C, ϵ = left_gauge(AC_expanded, alg.alg_gauge)
    B = _transpose_tail(ψ.AR[pos + 1])
    AR = _transpose_front(catcodomain(B, zeros(T, Vpert ← domain(B))))

    normalize && normalize!(C)
    ψ.AC[pos] = (AL, C)
    ψ.AC[pos + 1] = (C, AR)
    return ψ, ϵ
end

function gauge!(pos::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, envs, AC, alg::DMRG3S; normalize = true)
    Er = rightenv(envs, pos, ψ)
    Hi = H[pos]
    α = alg.noise
    T = promote_type(scalartype(ψ), scalartype(Hi))
    V = left_virtualspace(AC)
    combiner, Vpert = _get_combiner(T, V, left_virtualspace(Hi))

    @plansor pert[l; r s] := α * (combiner')[l; li lh] * AC[li, si; ri] * Hi[lh, s; si, rh] * Er[ri rh; r]

    AC = _transpose_tail(AC)
    AC_expanded = catcodomain(AC, pert)

    C, AR, ϵ = right_gauge(AC_expanded, alg.alg_gauge)
    B = ψ.AL[pos - 1]
    AL = catdomain(B, zeros(T, codomain(B) ← Vpert))

    normalize && normalize!(C)
    ψ.AC[pos] = (C, AR)
    ψ.AC[pos - 1] = (AL, C)
    return ψ, ϵ
end
