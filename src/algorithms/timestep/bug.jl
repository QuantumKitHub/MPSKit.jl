"""
$(TYPEDEF)

Single-site, symmetric second-order time-evolution algorithm for **finite** MPS, based on the
Basis-Update & Galerkin (BUG) integrator, an unconventional robust integrator for dynamical low-rank
approximation.

Unlike [`TDVP`](@ref), BUG advances both the basis (K-step) and the core (Galerkin C-step) tensors *forward* in time, with no backward-in-time substep.
This makes it a more natural choice for imaginary-time (dissipative) evolution, where the backward core step of the conventional projector-splitting [`TDVP`](@ref) can become unstable for large timesteps.

## Fields

$(TYPEDFIELDS)

## Algorithm

Each half-sweep visits every site in turn and, at each site,
(i) splits off the bond ahead of it (in the sweep direction) with `alg_gauge`, truncating it back to `trunc`,
(ii) evolves the connecting tensor over `dt/2`, and
(iii) augments the basis with the new directions discovered by the evolved tensor (old basis first, `[U₀ │ K₁]`, orthonormalized with `alg_orth`)

Notably, this last step does not include any truncation, and is meant to truncate the *previous* half-sweep's augmentation.
As a result, a truncation scheme `truncrank(D)` will result in a final MPS of dimension `2D`.
To restore a maximal dimension of `D`, apply [`changebonds`](@ref) with an [`SvdCut`](@ref) algorithm.

!!! note
    By default the state is not renormalized, as the (loss of) norm might accumulate useful information,
    such as the accumulated truncation error in real time, or the decaying weight in imaginary time.
    Pass `normalize = true` to `timestep`/`time_evolve` to renormalize after every half-sweep instead.

!!! tip
    Pass a `TimerOutputs.TimerOutput` as `timeroutput` to `timestep`/`timestep!` to obtain a
    breakdown of the time spent in the three steps above (`cut_bond`, `AC_integrate`, `augment`)
    over all local updates.

## References

* [Ceruti et al. BIT Numer. Math. 62 (2022)](@cite ceruti2022)
"""
struct BUG{A, O, G, F} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A

    "algorithm used to orthonormalize the augmented basis `[U₀ │ K₁]` after each local update"
    alg_orth::O

    "factorization used to gauge and truncate the bond ahead of each local update"
    alg_gauge::G

    "callback function applied after each iteration, of signature `finalize(t, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function BUG(;
        integrator = Defaults.alg_expsolve(), alg_orth = Defaults.alg_orth(),
        trunc = notrunc(), alg_svd = Defaults.alg_svd(),
        finalize = Defaults._finalize
    )
    alg_gauge = _build_inner_gauge(trunc, alg_svd, alg_orth)
    return BUG(integrator, alg_orth, alg_gauge, finalize)
end

# `ψ.AC[site]` first, the neighbour second: the lazy `CView` walk keys off what is already cached, and
# both reads must precede the installs, which move the gauge center across the bond.
function _cut_bond!(site::Int, ::Val{:right}, ψ, alg)
    AC₀ = ψ.AC[site]
    AR_next = ψ.AR[site + 1]

    # nothing changes without truncation, so `ψ` is left alone — which is also why the non-mutating
    # `left_orth` is required here: `AC₀` is still the center that gets evolved next
    if !_truncates(alg)
        AL_old, C₀ = left_orth(AC₀; alg)
        return AL_old, C₀, AR_next, zero(real(scalartype(ψ)))
    end

    # `Vᴴ` goes into the neighbour rather than into `C₀`, so the local update happens in the truncated
    # bond space; in-place, as the installs below invalidate the `AC` cache anyway
    AL_old, C₀, Vᴴ, ϵ = svd_trunc!(AC₀, alg)
    AR_next = _mul_front(Vᴴ, AR_next)                    # old right block, rotated into the cut bond
    ψ.AC[site] = (AL_old, C₀)
    ψ.AC[site + 1] = (C₀, AR_next)
    return AL_old, C₀, AR_next, ϵ
end

# mirror on the `_transpose_tail` form, which the augment step also works in
function _cut_bond!(site::Int, ::Val{:left}, ψ, alg)
    AC₀ = ψ.AC[site]
    AL_prev = ψ.AL[site - 1]

    if !_truncates(alg)
        C₀, AR_old_tail = right_orth(_transpose_tail(AC₀); alg)
        return AR_old_tail, C₀, AL_prev, zero(real(scalartype(ψ)))
    end

    U, C₀, AR_old_tail, ϵ = svd_trunc!(_transpose_tail(AC₀), alg)
    AL_prev = _mul_tail(AL_prev, U)                      # old left block, rotated into the cut bond
    ψ.AC[site] = (C₀, _transpose_front(AR_old_tail))
    ψ.AC[site - 1] = (AL_prev, C₀)
    return AR_old_tail, C₀, AL_prev, ϵ
end

# the site at which a sweep runs out of chain: no bond ahead to cut, so the local update is a plain
# evolve-and-install
_sweep_end(ψ, ::Val{:right}) = length(ψ)
_sweep_end(ψ, ::Val{:left}) = 1

# augment the old canonical basis at `site` with the new directions discovered by the evolved center
# (old first, and without truncation), and install it — together with the old center `C₀` transported
# into that basis and absorbed into the neighbouring site, which becomes the new gauge center
function _augment_basis!(site::Int, dir::Val{:right}, ψ, AL_old, AC, C₀, AR_next, alg_orth)
    AL_new, _ = left_orth!(catdomain(AL_old, AC); alg = alg_orth)
    transport = AL_new' * AL_old                         # old bond, in the augmented basis
    return set_canonical!(ψ, site, dir, AL_new, _mul_front(transport * C₀, AR_next))
end
function _augment_basis!(site::Int, dir::Val{:left}, ψ, AR_old_tail, AC, C₀, AL_prev, alg_orth)
    _, AR_new_tail = right_orth!(
        catcodomain(AR_old_tail, _transpose_tail(AC)); alg = alg_orth
    )
    transport = AR_old_tail * AR_new_tail'
    return set_canonical!(
        ψ, site, dir, _transpose_front(AR_new_tail), _mul_tail(AL_prev, C₀ * transport)
    )
end

function _evolve_center(site, ψ, H, alg::BUG, envs, t, h; imaginary_evolution)
    Heff = AC_hamiltonian(site, ψ, H, ψ, envs)
    return integrate(Heff, ψ.AC[site], t, h, alg.integrator; imaginary_evolution)
end

function local_update!(
        site, direction::Val, ψ, H, alg::BUG, envs, t, h;
        imaginary_evolution, normalize, timeroutput
    )
    # at the far end of the sweep there is no bond ahead to cut: evolve and finalize
    if site == _sweep_end(ψ, direction)
        AC = @timeit timeroutput "AC_integrate" _evolve_center(
            site, ψ, H, alg, envs, t, h; imaginary_evolution
        )
        normalize && normalize!(AC)
        ψ.AC[site] = AC
        return ψ
    end

    # 1. cut (and truncate) the bond ahead, before evolving
    A_old, C₀, neighbour, _ = @timeit timeroutput "cut_bond" _cut_bond!(
        site, direction, ψ, alg.alg_gauge
    )

    # 2. evolve the connecting tensor
    AC = @timeit timeroutput "AC_integrate" _evolve_center(
        site, ψ, H, alg, envs, t, h; imaginary_evolution
    )

    # 3. augment the basis (old first, no truncation here) and install it, together with the
    #    transported old center at the neighbouring site
    return @timeit timeroutput "augment" _augment_basis!(
        site, direction, ψ, A_old, AC, C₀, neighbour, alg.alg_orth
    )
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::BUG,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false, normalize::Bool = false,
        timeroutput::TimerOutput = DISABLED_TIMER
    )
    # symmetric 2nd-order: a dt/2 left→right half-sweep composed with its dt/2 mirror
    L = length(ψ)
    h = dt / 2

    # left→right half-sweep (root = last site): `ψ` carries the old center forward, site by site, so
    # `ψ.AR[site + 1]` is still the old (pre-sweep) right block when site `site` needs it
    @timeit timeroutput "half-sweep" for site in 1:L
        ψ = local_update!(
            site, Val(:right), ψ, H, alg, envs, t, h;
            imaginary_evolution, normalize, timeroutput
        )
    end

    # right→left half-sweep (root = first site), the mirror: sweep backward from `t + h`, with
    # `ψ.AL[site - 1]` the left block as the previous half-sweep left it
    @timeit timeroutput "half-sweep" for site in L:-1:1
        ψ = local_update!(
            site, Val(:left), ψ, H, alg, envs, t + h, h;
            imaginary_evolution, normalize, timeroutput
        )
    end

    return ψ, envs
end

# copying version
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
        alg::BUG, envs::AbstractMPSEnvironments...;
        imaginary_evolution::Bool = false, normalize::Bool = false, kwargs...
    )
    isreal = (scalartype(ψ) <: Real && !(imaginary_evolution && scalartype(H) <: Real))
    ψ′ = isreal ? complex(ψ) : copy(ψ)
    if length(envs) != 0 && isreal
        @warn "Currently cannot reuse real environments for complex evolution"
        envs′ = environments(ψ′, H, ψ′)
    elseif length(envs) == 1
        envs′ = only(envs)
    else
        @assert length(envs) == 0 "Invalid signature"
        envs′ = environments(ψ′, H, ψ′)
    end
    return timestep!(ψ′, H, time, timestep, alg, envs′; imaginary_evolution, normalize, kwargs...)
end
