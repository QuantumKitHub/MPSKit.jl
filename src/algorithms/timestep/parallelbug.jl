"""
$(TYPEDEF)

Single site MPS time-evolution algorithm based on the *parallel* Basis-Update & Galerkin (BUG)
integrator for tree tensor networks, specialized to the linear (`FiniteMPS`) tree.

Unlike the sequential [`BUG`](@ref), every local problem is solved from the **same frozen `t₀`
snapshot** (`ψ₀ = copy(ψ)` and its environments): there is no sweep and no sequential dependency
between the local integrations, so they are mutually independent and parallelizable. Rooting the
caterpillar at the last site, every local center `AC[i]` is evolved forward from `t₀`. A cheap
leaves→root assembly pass (the augmentation `A_τ` of Ceruti et al. 2024, Alg. 4) then augments
every bond with the new directions discovered by the frozen evolutions — old basis first,
`[U₀ │ Ũ₁]` — where the new directions at each bond are orthonormalized from the evolved center
*stacked with a first-order coupling block* on the new rows of the previous bond, so that
directions discovered deep in the chain propagate up toward the root. The interior tensors of the
augmented state are isometries (the amplitude and phase of the evolved centers is discarded with
the orthonormalization); the root tensor is the evolved center `AC[L]` augmented with the
projection of the full time derivative onto the new directions, so the amplitude and all
first-order updates enter exactly once, at the root. Finally an SVD sweep truncates the (at most
doubled) bonds back down.
Like [`BUG`](@ref), it advances every tensor *forward* in time (no backward substep), which suits
imaginary-time / dissipative evolution.

The integrator is **first-order** in time (the discarded "new–new" coupling corners are `O(dt²)`),
in contrast to the second-order symmetrized sequential [`BUG`](@ref). The truncation tolerance of
`trscheme` maps to the BUG tolerance `ϑ`; because the global error accumulates as `c·n·ϑ` over `n`
steps, scale `ϑ` with `dt` for a fixed target accuracy. Any truncating `trscheme` (e.g.
`truncerror`) makes the bond dimension grow and shrink to track the entanglement of the evolving
state; the default `notrunc()` restores the pre-step virtual spaces after every step (fixed-rank
parallel BUG).

!!! warning "Experimental"
    This integrator is **work in progress**: the parallel-in-time structure is implemented (all
    local solves read from one frozen snapshot), but the local solves are currently executed
    serially, and the step-rejection criterion of Ceruti et al. (2024) is not yet implemented.
    The API and behaviour may change.

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

    # remember the pre-step virtual spaces for the fixed-rank (`notrunc`) restore
    Vs = [right_virtualspace(ψ, b) for b in 1:(L - 1)]

    ϕ = _pbug_assemble(ψ, H, t, dt, alg; imaginary_evolution)
    _pbug_truncate!(ϕ, alg, Vs; normalize = imaginary_evolution)

    # mutate `ψ` in place to become the assembled state (the generic loop reuses the object)
    _pbug_overwrite!(ψ, ϕ)
    return ψ, environments(ψ, H, ψ)
end

# Assemble the augmented (pre-truncation) parallel-BUG state from a single frozen `t₀` snapshot,
# following Ceruti et al. 2024 (arXiv:2412.00858) Alg. 1-4 specialized to the caterpillar tree
# rooted at site `L`. Two phases:
#
# 1. Galerkin-evolve every local center `AC[i]` from the frozen snapshot (the K-steps of
#    Alg. 2/3, on the amplitude-weighted subtree objects `Y_τ⁰ = U_τ⁰S_τ⁰`). These `L` local
#    solves are mutually independent, which is the parallel-in-time structure of the integrator.
# 2. A leaves→root augmentation pass (Alg. 4): at bond `i` the new directions `Ũᵢ` are
#    orthonormalized against the (zero-padded) old isometry from the evolved center *stacked with
#    the first-order coupling block* `C̃ᵢ = dt′·⟨Ũᵢ₋₁|H|ψ₀⟩-derivative of AC[i]` on the new rows of
#    the previous bond — this reconciliation is what lets directions discovered deep in the chain
#    propagate to the root. Only the range of these stacks is kept (their amplitude and phase are
#    discarded with the R-factor), so the interior site tensors of the augmented state are the
#    isometries `[old │ Ũᵢ]`, and all first-order (and amplitude) content is carried by the root
#    tensor `[C̄_L(t₁); C̃_L]`, whose coupling row projects the full derivative of `ψ₀` onto the
#    new directions. The "new-new" corners are implicitly zero, which is what makes the integrator
#    first order.
function _pbug_assemble(ψ, H, t, dt, alg::ParallelBUG; imaginary_evolution::Bool = false)
    L = length(ψ)
    ψ.AC[L]                                     # gauge to the root
    ψ₀ = copy(ψ)
    envs₀ = environments(ψ₀, H, ψ₀)
    dt′ = imaginary_evolution ? -dt : -im * dt

    # phase 1: frozen-snapshot Galerkin evolutions (independent local solves). The interior
    # K-steps evolve the amplitude-weighted centers `AC[i]` (the paper's `Y_τ⁰ = U_τ⁰S_τ⁰`); only
    # their range is kept, so the amplitude they carry is discarded in the orthonormalization.
    Cevo = map(1:(L - 1)) do i
        return integrate(
            AC_hamiltonian(i, ψ₀, H, ψ₀, envs₀), ψ₀.AC[i], t, dt, alg.integrator;
            imaginary_evolution
        )
    end
    C̄L = integrate(
        AC_hamiltonian(L, ψ₀, H, ψ₀, envs₀), ψ₀.AC[L], t, dt, alg.integrator; imaginary_evolution
    )

    # phase 2: leaves→root augmentation sweep, threading the mixed ⟨augmented|H|old⟩ environments
    As = Vector{Any}(undef, L)
    GLmix = _pbug_mixedenv_init(H, envs₀, ψ₀)   # full augmented rows (trivial at the left edge)
    local GLnew                                 # new-direction rows only
    for i in 1:(L - 1)
        if i == 1
            C⁰, Ĉ¹ = ψ₀.AL[1], Cevo[1]
        else
            # first-order coupling block on the new rows of bond i-1
            C̃ = scale(_pbug_coupling_hamiltonian(GLnew, H, i, envs₀, ψ₀)(ψ₀.AC[i]), dt′)
            C⁰ = _pbug_stack_child(ψ₀.AL[i], zerovector!(similar(C̃)))
            Ĉ¹ = _pbug_stack_child(Cevo[i], C̃)
        end
        Ũ, = _pbug_newdirs(C⁰, Ĉ¹, alg.alg_orth)
        As[i] = catdomain(C⁰, Ũ)
        GLnew = _pbug_mixedenv_step(GLmix, H, i, ψ₀.AL[i], Ũ)
        i == L - 1 && break                     # the full mixed environment is no longer needed
        GLmix = _pbug_mixedenv_step(GLmix, H, i, ψ₀.AL[i], As[i])
    end
    # root: the amplitude is carried once, by the evolved center and its coupling row
    C̃L = scale(_pbug_coupling_hamiltonian(GLnew, H, L, envs₀, ψ₀)(ψ₀.AC[L]), dt′)
    As[L] = _pbug_stack_child(C̄L, C̃L)

    return FiniteMPS(
        convert(Vector{typeof(As[1])}, As); overwrite = true, normalize = imaginary_evolution
    )
end

# Mixed ⟨augmented|H|old⟩ left environments: initial (trivial-bond) environment and one-site
# transfer step, with the same operator dispatch as the effective-Hamiltonian constructors.
_pbug_mixedenv_init(H, envs, ψ₀) = leftenv(envs, 1, ψ₀)
function _pbug_mixedenv_init(H::MultipliedOperator, envs, ψ₀)
    return _pbug_mixedenv_init(H.op, envs, ψ₀)
end
function _pbug_mixedenv_init(H::LazySum, envs::MultipleEnvironments, ψ₀)
    return map((o, e) -> _pbug_mixedenv_init(o, e, ψ₀), H.ops, envs.envs)
end

_pbug_mixedenv_step(GL, H, i, above, below) = GL * TransferMatrix(above, H[i], below)
function _pbug_mixedenv_step(GL, H::MultipliedOperator, i, above, below)
    return _pbug_mixedenv_step(GL, H.op, i, above, below)
end
function _pbug_mixedenv_step(GLs::Vector, H::LazySum, i, above, below)
    return map((gl, o) -> _pbug_mixedenv_step(gl, o, i, above, below), GLs, H.ops)
end

# One-site effective derivative with the mixed new-direction rows as left environment and the
# frozen old right environment: applied to the center `AC[i]` this yields the first-order coupling
# block `C̃ᵢ` (Alg. 4's `h·F(Y_τ⁰) ×ⱼ U⁰* ×ᵢ Ũ¹*`).
function _pbug_coupling_hamiltonian(GL, H, i, envs, ψ₀)
    return MPO_AC_Hamiltonian(GL, H[i], rightenv(envs, i, ψ₀))
end
function _pbug_coupling_hamiltonian(GL, H::MultipliedOperator, i, envs, ψ₀)
    return MultipliedOperator(_pbug_coupling_hamiltonian(GL, H.op, i, envs, ψ₀), H.f)
end
function _pbug_coupling_hamiltonian(GLs::Vector, H::LazySum, i, envs::MultipleEnvironments, ψ₀)
    Hs = map((gl, o, e) -> _pbug_coupling_hamiltonian(gl, o, i, e, ψ₀), GLs, H.ops, envs.envs)
    elT = Union{D, MultipliedOperator{D}} where {D <: DerivativeOperator}
    return LazySum{elT}(Hs)
end

# new bond directions: the component of the evolved (stacked) candidate orthogonal to the old
# basis, re-orthonormalized. Returns the isometry `Ũ` and the new bond space.
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
# `notrunc()` restores the pre-step virtual space of every bond (fixed-rank parallel BUG).
# Environments self-heal lazily for the changed bonds.
function _pbug_truncate!(ϕ, alg::ParallelBUG, Vs; normalize::Bool = false)
    if _pbug_truncates(alg)
        changebonds!(ϕ, SvdCut(; trscheme = alg.trscheme, alg_svd = alg.alg_svd); normalize)
    else
        # per-bond fixed-rank restore (mirrors the `SvdCut` sweep with a per-bond `trscheme`)
        for i in (length(ϕ) - 1):-1:1
            U, S, Vᴴ = svd_trunc(ϕ.C[i]; trunc = truncspace(Vs[i]), alg = alg.alg_svd)
            ϕ.AC[i] = (ϕ.AL[i] * U, S)
            ϕ.AC[i + 1] = (S, _transpose_front(Vᴴ * _transpose_tail(ϕ.AR[i + 1])))
        end
        normalize && normalize!(ϕ)
    end
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
