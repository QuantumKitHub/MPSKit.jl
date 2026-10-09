"""
$(TYPEDEF)

Abstract supertype for the different flavours of dynamical DMRG.
"""
abstract type DDMRG_Flavour end

"""
$(TYPEDEF)

A dynamical DMRG method for calculating dynamical properties and excited states, based on a
variational principle for dynamical correlation functions.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`propagator`](@ref).

# References

* [Jeckelmann. Phys. Rev. B 66 (2002)](@cite jeckelmann2002)
"""
@kwdef struct DynamicalDMRG{F <: DDMRG_Flavour, S, B} <: Algorithm
    "flavour of the algorithm to use, either of type [`NaiveInvert`](@ref) or [`Jeckelmann`](@ref)"
    flavour::F = NaiveInvert()
    "algorithm used for the linear solvers"
    solver::S = Defaults.linearsolver
    "convergence tolerance on the largest change of a center tensor over a sweep"
    tol::Float64 = Defaults.tol * 10
    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter
    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity
    "backend for tensor contractions and index manipulations"
    backend::B = Defaults.backend()
end

IterativeLoggers.IterLog(::DynamicalDMRG) = IterLog("DDMRG")

"""
    propagator(ψ₀::AbstractFiniteMPS, z::Number, H::MPOHamiltonian, alg::DynamicalDMRG; init = copy(ψ₀)) -> (g, ψ)

Calculate the action of the propagator ``\\frac{1}{z - H}|ψ₀⟩`` using the dynamical DMRG
algorithm.

# Returns

- `g`: approximation of the propagator matrix element ``⟨ψ₀|\\frac{1}{z - H}|ψ₀⟩``
- `ψ`: MPS approximation of ``\\frac{1}{z - H}|ψ₀⟩``
"""
function propagator end

"""
$(TYPEDEF)

An alternative approach to the dynamical DMRG algorithm, without quadratic terms but with a
less controlled approximation.
This algorithm minimizes the following cost function
```math
⟨ψ|(z - H)|ψ⟩ - ⟨ψ|ψ₀⟩ - ⟨ψ₀|ψ⟩
```

Returns the approximation of ``⟨ψ₀|\\frac{1}{z - H}|ψ₀⟩`` and ``\\frac{1}{z - H}|ψ₀⟩``.

# See also

[`Jeckelmann`](@ref) for the original approach.
"""
struct NaiveInvert <: DDMRG_Flavour end

# Internal state of the dynamical DMRG sweeps, which update `mps` in place towards the
# propagator applied to `target`; `ϵ` is the largest change of a center tensor over the last sweep
struct DDMRGState{S, T, Z, O, E, A}
    mps::S
    target::T
    z::Z
    operator::O
    envs::E
    iter::Int
    ϵ::Float64
    allocator::A
end

# dynamical DMRG sweeps over the sites like single-site DMRG
_sweep_ranges(::DynamicalDMRG, ψ) = (1:(length(ψ) - 1), length(ψ):-1:2)

function Base.iterate(it::IterativeSolver{<:DynamicalDMRG}, state::DDMRGState)
    iter = state.iter + 1
    state = DDMRGState(
        state.mps, state.target, state.z, state.operator, state.envs, state.iter, 0.0,
        state.allocator,
    )
    state = sweep!(it, state, Val(:right), iter)
    state = sweep!(it, state, Val(:left), iter)
    it.state = DDMRGState(
        state.mps, state.target, state.z, state.operator, state.envs, iter, state.ϵ,
        state.allocator,
    )
    return (state.mps, state.ϵ), it.state
end

function _propagator_sweeps!(alg::DynamicalDMRG, state::DDMRGState)
    log = IterLog(alg)
    it = IterativeSolver(alg, state)

    with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, it.ϵ)
        for (_, ϵ) in Iterators.take(it, alg.maxiter)
            if ϵ <= alg.tol
                @log_convergence logfinish!(log, it.iter, ϵ)
                break
            elseif it.iter == alg.maxiter
                @log_nonconvergence logcancel!(log, it.iter, ϵ)
            else
                @log_iteration logiter!(log, it.iter, ϵ)
            end
        end
    end

    return it.state
end

function propagator(
        A::AbstractFiniteMPS, z::Number, H,
        alg::DynamicalDMRG{NaiveInvert}; init = copy(A)
    )
    allocator = default_allocator(A, SerialScheduler())
    h_envs = environments(init, H, init) # environments for h
    mixedenvs = environments(init, A) # environments for <init | A>

    state = DDMRGState(init, A, z, H, (h_envs, mixedenvs), 0, 2 * alg.tol, allocator)
    _propagator_sweeps!(alg, state)

    return dot(A, init), init
end

function sweep!(
        it::IterativeSolver{<:DynamicalDMRG{NaiveInvert}}, state::DDMRGState, direction, iter
    )
    alg = it.alg
    (; mps, target, z, operator, allocator) = state
    init, A, H = mps, target, operator
    h_envs, mixedenvs = state.envs
    fwd, bwd = _sweep_ranges(alg, A)
    ϵ = state.ϵ

    for i in (direction === Val(:right) ? fwd : bwd)
        tos = AC_projection(i, init, A, mixedenvs; alg.backend, allocator)

        H_AC = AC_hamiltonian(i, init, H, init, h_envs; alg.backend, allocator)
        AC = init.AC[i]
        AC′, convhist = linsolve(H_AC, -tos, AC, alg.solver, -z, one(z))

        ϵ = max(ϵ, norm(AC′ - AC))
        init.AC[i] = AC′

        convhist.converged == 0 &&
            @warn "propagator ($i) failed to converge: normres = $(convhist.normres)"
    end

    return DDMRGState(
        init, A, z, operator, state.envs, state.iter, ϵ, allocator,
    )
end

"""
$(TYPEDEF)

The original flavour of dynamical DMRG, which minimizes functional (14) from Jeckelmann2002.
Writing ``ω = \\mathrm{Re}(z)`` and ``η = \\mathrm{Im}(z)``, this is
```math
W(ψ) = ⟨ψ|(ω - H)^2 + η^2|ψ⟩ + η(⟨ψ₀|ψ⟩ + ⟨ψ|ψ₀⟩)
```
which attains its minimum at
```math
((ω - H)^2 + η^2)|ψ⟩ = -η|ψ₀⟩
```

Together with equation (11) from that same paper we can determine the full propagator
``\\frac{1}{z - H}|ψ₀⟩``.

Returns the approximation of ``⟨ψ₀|\\frac{1}{z - H}|ψ₀⟩`` and ``\\frac{1}{z - H}|ψ₀⟩``.

# See also

[`NaiveInvert`](@ref) for a less costly but less accurate alternative.

# References

* [Jeckelmann. Phys. Rev. B 66 (2002)](@cite jeckelmann2002)
"""
struct Jeckelmann <: DDMRG_Flavour end

function propagator(
        A::AbstractFiniteMPS, z::Number, H,
        alg::DynamicalDMRG{Jeckelmann}; init = copy(A)
    )
    allocator = default_allocator(A, SerialScheduler())
    ω = real(z)
    η = imag(z)

    envs1 = environments(init, H, init) # environments for h
    H2, envs2 = squaredenvs(init, H, envs1) # environments for h^2
    mixedenvs = environments(init, A) # environments for <init | A>

    state = DDMRGState(init, A, z, (H, H2), (envs1, envs2, mixedenvs), 0, 2 * alg.tol, allocator)
    _propagator_sweeps!(alg, state)

    a = dot(AC_projection(1, init, A, mixedenvs; alg.backend, allocator), init.AC[1])
    cb = leftenv(envs1, 1, A) * TransferMatrix(init.AL, H[1:length(A.AL)], A.AL)
    b = zero(a)
    for i in 1:length(cb)
        b += @plansor cb[i][1 2; 3] * init.C[end][3; 4] *
            rightenv(envs1, length(A), A)[i][4 2; 5] * conj(A.C[end][1; 5])
    end

    v = b / η - ω / η * a + 1im * a
    return v, init
end

function sweep!(
        it::IterativeSolver{<:DynamicalDMRG{Jeckelmann}}, state::DDMRGState, direction, iter
    )
    alg = it.alg
    (; mps, target, z, allocator) = state
    init, A = mps, target
    H, H2 = state.operator
    envs1, envs2, mixedenvs = state.envs
    ω = real(z)
    η = imag(z)
    fwd, bwd = _sweep_ranges(alg, A)
    ϵ = state.ϵ

    for i in (direction === Val(:right) ? fwd : bwd)
        tos = AC_projection(i, init, A, mixedenvs; alg.backend, allocator)
        H1_AC = AC_hamiltonian(i, init, H, init, envs1; alg.backend, allocator)
        H2_AC = AC_hamiltonian(i, init, H2, init, envs2; alg.backend, allocator)
        H_AC = LinearCombination((H1_AC, H2_AC), (-2 * ω, 1))
        AC′, convhist = linsolve(H_AC, -η * tos, init.AC[i], alg.solver, abs2(z), 1)

        ϵ = max(ϵ, norm(AC′ - init.AC[i]))
        init.AC[i] = AC′

        convhist.converged == 0 &&
            @warn "propagator ($i) failed to converge: normres $(convhist.normres)"
    end

    return DDMRGState(
        init, A, z, state.operator, state.envs, state.iter, ϵ, allocator,
    )
end

function squaredenvs(
        state::AbstractFiniteMPS, H, envs = environments(state, H, state)
    )
    H² = conj(H) * H
    L = length(state)

    # impose the correct boundary conditions (important for WindowMPS)
    leftstart = _contract_leftenv²(leftenv(envs, 1, state), leftenv(envs, 1, state))
    rightstart = _contract_rightenv²(rightenv(envs, L, state), rightenv(envs, L, state))

    # to construct the squared caches we will first initialize environments
    # then make all data invalid so it will be recalculated
    envs² = environments(state, H², state; leftstart, rightstart)
    for i in 1:L
        poison!(envs², i)
    end

    return H², envs²
end

function _contract_leftenv²(GL_top, GL_bot)
    V_mid = space(GL_bot, 2)' ⊗ space(GL_top, 2)
    F = isomorphism(storagetype(GL_top), fuse(V_mid)' ← V_mid)
    return @plansor GL[-1 -2; -3] := GL_top[1 3; -3] * conj(GL_bot[1 2; -1]) * F[-2; 2 3]
end

function _contract_rightenv²(GR_top, GR_bot)
    V_mid = space(GR_top, 2) ⊗ space(GR_bot, 2)'
    F = isomorphism(storagetype(GR_top), fuse(V_mid) ← V_mid)
    return @plansor GR[-1 -2; -3] := GR_top[-1 2; 1] * conj(GR_bot[-3 3; 1]) * F[-2; 2 3]
end
