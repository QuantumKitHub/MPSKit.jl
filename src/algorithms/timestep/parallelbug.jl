"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the *parallel* Basis-Update & Galerkin (BUG)
integrator for tree tensor networks, specialized to the linear (`FiniteMPS`) tree.

Unlike the sequential [`BUG`](@ref), every local problem is solved from the **same frozen `t₀`
snapshot** (`ψ₀ = copy(ψ)` and its environments): there is no sweep and no sequential dependency
between the local integrations, so they are mutually independent and parallelizable. Rooting the
caterpillar at the last site, the amplitude is carried by the root center `AC[L]` while the interior
connecting tensors are *isometries* `AL[i]`; each is evolved forward from `t₀`. Every bond is then
augmented with the new directions discovered by the frozen evolution (`[U₀ │ Ũ₁]`, old basis first)
together with a first-order coupling block, and truncated back down by an SVD sweep. Like
[`BUG`](@ref), it advances every tensor *forward* in time (no backward substep), which suits
imaginary-time / dissipative evolution.

The truncation tolerance of `trscheme` maps to the BUG tolerance `ϑ`; because the global error
accumulates as `c·n·ϑ` over `n` steps, scale `ϑ` with `dt` for a fixed target accuracy. Any
truncating `trscheme` (e.g. `truncerror`) makes the bond dimension grow and shrink to track the
entanglement of the evolving state.

!!! warning "Experimental"
    This integrator is **work in progress**. It reproduces the exact matrix parallel-BUG step for
    two sites, conserves energy / the eigenstate phase exactly, and grows bonds adaptively, but the
    coupling-block reconciliation (the `M = Û'U₀` reprojection of Ceruti et al. 2024, Alg. 4) is not
    yet complete, so it does **not** yet attain the documented first-order accuracy for `L > 2`. See
    `research/PARALLELBUG_design.md` and `research/PARALLELBUG_STATUS.md`. The API and behaviour may
    change.

!!! note
    Real-time evolution does not normalize the resulting state: neither the augmentation nor the
    truncation normalizes, so the state norm reflects the accumulated truncation error.
    Imaginary-time evolution renormalizes after every step, similar to a ground-state search.

## Fields

$(TYPEDFIELDS)

## References

* Ceruti, Kusch & Lubich, *A parallel rank-adaptive integrator for dynamical low-rank
  approximation*, SIAM J. Sci. Comput. **46** (2024).
* Ceruti, Kusch, Lubich & Sulz, *A parallel Basis Update and Galerkin integrator for tree tensor
  networks*, arXiv:2412.00858 (2024).
"""
struct ParallelBUG{A, O, T, S, F} <: Algorithm
    "algorithm used in the exponential solvers"
    integrator::A

    "tolerance for gauging algorithm"
    tolgauge::Float64

    "maximal amount of iterations for gauging algorithm"
    gaugemaxiter::Int

    "algorithm used to re-orthonormalize the basis after each local update"
    alg_orth::O

    "truncation scheme used to cut the augmented bonds back down"
    trscheme::T

    "algorithm used for the singular value decomposition"
    alg_svd::S

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function ParallelBUG(;
        integrator = Defaults.alg_expsolve(), tolgauge = Defaults.tolgauge,
        gaugemaxiter = Defaults.maxiter, alg_orth = Defaults.alg_orth(),
        trscheme = notrunc(), alg_svd = Defaults.alg_svd(),
        finalize = Defaults._finalize
    )
    return ParallelBUG(integrator, tolgauge, gaugemaxiter, alg_orth, trscheme, alg_svd, finalize)
end

function timestep!(
        ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::ParallelBUG,
        envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        imaginary_evolution::Bool = false
    )
    L = length(ψ)
    if L == 1                                   # single site: a plain forward center step
        AC = integrate(
            AC_hamiltonian(1, ψ, H, ψ, envs), ψ.AC[1], t, dt, alg.integrator; imaginary_evolution
        )
        imaginary_evolution && normalize!(AC)
        ψ.AC[1] = AC
        return ψ, envs
    end

    # remember the pre-step bond dimensions for the fixed-rank (`notrunc`) restore
    Ds = [dim(right_virtualspace(ψ, b)) for b in 1:(L - 1)]

    ϕ = _pbug_assemble(ψ, H, t, dt, alg; imaginary_evolution)
    _pbug_truncate!(ϕ, alg, Ds; normalize = imaginary_evolution)

    # mutate `ψ` in place to become the assembled state (the generic loop reuses the object)
    _pbug_overwrite!(ψ, ϕ)
    return ψ, environments(ψ, H, ψ)
end

# Assemble the augmented (pre-truncation) parallel-BUG state from a single frozen `t₀` snapshot.
# Root the caterpillar at site `L`; only the root center `AC[L]` carries amplitude, the interior
# connecting tensors are the isometries `AL[i]`. Each is evolved forward from the frozen snapshot;
# the interior evolutions define the new bond directions `Ũ`, and first-order coupling blocks connect
# them (the "new–new" corners are left at zero — this is what makes the integrator first order).
function _pbug_assemble(ψ, H, t, dt, alg::ParallelBUG; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[L]                                     # gauge to the root
    ψ₀ = copy(ψ)
    envs₀ = environments(ψ₀, H, ψ₀)
    dt′ = imaginary_evolution ? -dt : -im * dt

    # interior isometry evolutions → new bond directions `Ũ_i` on bond `i`
    Us = Vector{Any}(undef, L - 1)
    Ns = Vector{Any}(undef, L - 1)
    for i in 1:(L - 1)
        Cevo = integrate(
            AC_hamiltonian(i, ψ₀, H, ψ₀, envs₀), ψ₀.AL[i], t, dt, alg.integrator; imaginary_evolution
        )
        Us[i], Ns[i] = _pbug_newdirs(ψ₀.AL[i], Cevo, alg.alg_orth)
    end
    # root center evolution (carries the amplitude / eigenstate phase, once)
    CbarL = integrate(
        AC_hamiltonian(L, ψ₀, H, ψ₀, envs₀), ψ₀.AC[L], t, dt, alg.integrator; imaginary_evolution
    )

    # first-order coupling blocks C̃_i : Ñ_{i-1} ⊗ P_i ← V_i, from the frozen two-site derivative
    Ctil = Vector{Any}(undef, L)
    for i in 2:L
        T2 = _transpose_front(ψ₀.AL[i - 1]) * _transpose_tail(ψ₀.AC[i])
        F2 = AC2_hamiltonian(i - 1, ψ₀, H, ψ₀, envs₀) * T2
        Ctil[i] = scale(_transpose_front(Us[i - 1]' * F2), dt′)
    end

    # stack the augmented site tensors (old block first, coupling in the new-child rows, zero corner)
    As = Vector{Any}(undef, L)
    As[1] = catdomain(copy(ψ₀.AL[1]), Us[1])
    for i in 2:(L - 1)
        top = catdomain(copy(ψ₀.AL[i]), Us[i])
        Z = zerovector!(similar(Ctil[i], codomain(Ctil[i]) ← Ns[i]))
        bot = catdomain(Ctil[i], Z)
        As[i] = _pbug_stack_child(top, bot)
    end
    As[L] = _pbug_stack_child(CbarL, Ctil[L])

    return FiniteMPS(
        convert(Vector{typeof(As[1])}, As); overwrite = true, normalize = imaginary_evolution
    )
end

# new bond directions: the component of the evolved (interior) isometry orthogonal to the old basis,
# re-orthonormalized. Returns the isometry `Ũ` and the new bond space.
function _pbug_newdirs(AL, Cevo, alg_orth = Defaults.alg_orth())
    N = left_null(AL)
    g = N' * Cevo
    Q, _ = left_orth(g; alg = alg_orth)
    Ũ = N * Q
    return Ũ, domain(Q)
end

# stack two MPS tensors along the child (left-virtual) bond: `[top; bot]`, doubling that bond
_pbug_stack_child(top, bot) =
    _transpose_front(catcodomain(_transpose_tail(top), _transpose_tail(bot)))

_pbug_truncates(alg::ParallelBUG) = !(alg.trscheme isa MatrixAlgebraKit.NoTruncation)

# Cut the augmented bonds back down. A truncating `trscheme` selects rank-adaptivity; the default
# `notrunc()` restores the pre-step bond dimensions (fixed-rank parallel BUG). Environments
# self-heal lazily for the changed bonds.
function _pbug_truncate!(ϕ, alg::ParallelBUG, Ds; normalize::Bool = false)
    trscheme = _pbug_truncates(alg) ? alg.trscheme : truncrank(maximum(Ds))
    changebonds!(ϕ, SvdCut(; trscheme, alg_svd = alg.alg_svd); normalize)
    return ϕ
end

# overwrite the internal representation of `ψ` with that of `ϕ` (same length, possibly new bonds)
function _pbug_overwrite!(ψ::FiniteMPS, ϕ::FiniteMPS)
    for f in (:ALs, :ARs, :ACs, :Cs)
        copyto!(getfield(ψ, f), getfield(ϕ, f))
    end
    return ψ
end

# copying version (mirrors `bug.jl`/`tdvp.jl`)
function timestep(
        ψ::AbstractFiniteMPS, H, time::Number, timestep::Number,
        alg::ParallelBUG, envs::AbstractMPSEnvironments...;
        imaginary_evolution::Bool = false, kwargs...
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
    return timestep!(ψ′, H, time, timestep, alg, envs′; imaginary_evolution, kwargs...)
end
