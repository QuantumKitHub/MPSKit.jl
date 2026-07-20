"""
$(TYPEDEF)

Single-site, symmetric second-order time-evolution algorithm for **finite** MPS, based on the
Basis-Update & Galerkin (BUG) integrator, an unconventional robust integrator for dynamical low-rank
approximation.

Unlike [`TDVP`](@ref), BUG advances both the basis (K-step) and the core (Galerkin C-step) tensors
*forward* in time, with no backward-in-time substep. This makes it a natural choice for
imaginary-time / dissipative evolution, where the backward core step of TDVP can become unstable.

Each half-sweep augments every bond with the new directions discovered by the evolved connecting
tensor (old basis first, `[U₀ │ K₁]`) and truncates back down to the tolerance of `trscheme` in the
same orthonormalization step. The truncation is folded into the augment orth (a truncated SVD of the
stacked basis), so the old directions — whose orthonormal columns carry singular value `≥ 1` — are
always kept and only newly-appended directions are cut. The bond therefore *grows* to track the
entanglement under a `truncerror` tolerance; a hard rank cap (`truncrank`) can additionally shrink it.

!!! warning
    With `trscheme = notrunc()` the augmented basis is kept at full rank every half-sweep, so the
    bond grows unboundedly (up to the local Hilbert-space dimension) — this is **not** a fixed-rank
    integrator. To cap the bond dimension at `D`, pass `trscheme = truncrank(D)`: because the old
    directions carry singular value `≥ 1`, a rank-`D` cut keeps them and drops the newly appended
    directions, so a bond already at `D` stays there (a fixed-rank-`D` step). A bond currently below
    `D` can still grow up to `D` as new directions are admitted.

!!! note
    By default the state is not renormalized, so the norm keeps useful information (the
    accumulated truncation error in real time, or the decaying weight in imaginary time).
    Pass `normalize = true` to `timestep`/`time_evolve` to renormalize after every half-sweep
    instead. This is independent of `imaginary_evolution`.

## Fields

$(TYPEDFIELDS)

## References

* [Ceruti et al. BIT Numer. Math. 62 (2022)](@cite ceruti2022)
"""
struct BUG{A, O, G, F} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A

    "algorithm used to re-orthonormalize the basis after each local update"
    alg_orth::O

    "factorization used for the in-sweep augment gauge: a truncated SVD (`alg_svd` with `trscheme`)"
    alg_gauge::G

    "callback function applied after each iteration, of signature `finalize(t, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function BUG(;
        integrator = Defaults.alg_expsolve(), alg_orth = Defaults.alg_orth(),
        trscheme = notrunc(), alg_svd = Defaults.alg_svd(),
        finalize = Defaults._finalize
    )
    alg_gauge = MatrixAlgebraKit.TruncatedAlgorithm(alg_svd, trscheme)
    return BUG(integrator, alg_orth, alg_gauge, finalize)
end

# left→right BUG update at `site`: evolve the connecting tensor, old-first augment + truncate the
# basis, and return the `transport` and center `AC_old` advanced to `site + 1` (root = last site).
function local_update!(
        site, ::Val{:right}, ψ, H, alg::BUG, envs, t, h, transport, AC_old, ARs;
        imaginary_evolution, normalize
    )
    # 1. Transport the old AC to the new frame and evolve
    AC₀ = _mul_front(transport, AC_old)
    AC = integrate(
        AC_hamiltonian(site, ψ, H, ψ, envs), AC₀,
        t, h, alg.integrator; imaginary_evolution
    )

    # If end of chain, finalize
    if site == length(ψ)
        normalize && normalize!(AC)
        ψ.AC[site] = AC
        return ψ, transport, AC_old
    end

    # 2. move gauge right
    oldbasis, C₀ = left_gauge(AC₀, alg.alg_orth)                        # old AL - C in new frame
    AC_next = _mul_front(C₀, ARs[site + 1])


    # 3. reproject onto new basis
    newbasis, _ = left_gauge(catdomain(oldbasis, AC), alg.alg_gauge)    # augmented basis
    newbond = newbasis' * AC
    ψ.AC[site] = (newbasis, newbond)
    new_transport = newbasis' * oldbasis

    return ψ, new_transport, AC_next
end

# mirror of the left→right update on the `_transpose_tail` form; `transport` and `AC_old` are returned
# advanced to `site - 1` (root = first site).
function local_update!(
        site, ::Val{:left}, ψ, H, alg::BUG, envs, t, h, transport, AC_old, ALs;
        imaginary_evolution, normalize
    )
    # 1. Transport the old AC to the new frame and evolve
    AC₀ = AC_old * transport
    AC = integrate(
        AC_hamiltonian(site, ψ, H, ψ, envs), AC₀,
        t, h, alg.integrator; imaginary_evolution
    )

    # If start of chain, finalize
    if site == 1
        normalize && normalize!(AC)
        ψ.AC[site] = AC
        return ψ, transport, AC_old
    end

    # 2. move gauge left
    C₀, oldbasis = right_gauge(AC₀, alg.alg_orth)              # old C - AR in new frame
    AC_next = _mul_tail(ALs[site - 1], C₀)

    # 3. reproject onto new basis
    _, newbasis_tail = right_orth(
        catcodomain(_transpose_tail(oldbasis), _transpose_tail(AC));
        alg = MatrixAlgebraKit.RightOrthViaSVD(alg.alg_gauge)   # augmented basis
    )
    newbasis = _transpose_front(newbasis_tail)
    newbond = _transpose_tail(AC) * _transpose_tail(newbasis)'
    ψ.AC[site] = (newbond, newbasis)
    new_transport = _transpose_tail(oldbasis) * _transpose_tail(newbasis)'

    return ψ, new_transport, AC_next
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::BUG,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false, normalize::Bool = false
    )
    # symmetric 2nd-order: a dt/2 left→right half-sweep composed with its dt/2 mirror
    L = length(ψ)
    h = dt / 2

    # left→right half-sweep (root = last site): freeze the bases as `[AC[1], AR[2], …, AR[L]]` and
    # carry `AC_old`/`transport` forward
    ARs = ψ.AR[2:end]
    pushfirst!(ARs, ψ.AC[1])
    transport = isomorphism(scalartype(ψ), left_virtualspace(ψ, 1) ← left_virtualspace(ψ, 1))
    AC_old = ARs[1]
    for site in 1:L
        ψ, transport, AC_old = local_update!(
            site, Val(:right), ψ, H, alg, envs, t, h, transport, AC_old, ARs;
            imaginary_evolution, normalize
        )
    end

    # right→left half-sweep (root = first site), the mirror: freeze `[AL[1], …, AL[L-1], AC[L]]` and
    # carry `AC_old`/`transport` backward (starting from `t + h`)
    ALs = ψ.AL[1:(L - 1)]
    push!(ALs, ψ.AC[L])
    transport = isomorphism(scalartype(ψ), right_virtualspace(ψ, L) ← right_virtualspace(ψ, L))
    AC_old = ALs[L]
    for site in L:-1:1
        ψ, transport, AC_old = local_update!(
            site, Val(:left), ψ, H, alg, envs, t + h, h, transport, AC_old, ALs;
            imaginary_evolution, normalize
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
    isreal = (scalartype(ψ) <: Real && !imaginary_evolution)
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
