"""
$(TYPEDEF)

Abstract supertype for the different flavours of dynamical DMRG.
"""
abstract type DDMRG_Flavour end

"""
$(TYPEDEF)

A dynamical DMRG method for calculating dynamical properties and excited states, based on a
variational principle for dynamical correlation functions.

## Fields

$(TYPEDFIELDS)

## References

* [Jeckelmann. Phys. Rev. B 66 (2002)](@cite jeckelmann2002)
"""
@kwdef struct DynamicalDMRG{F<:DDMRG_Flavour,S} <: Algorithm
    "flavour of the algorithm to use, either of type [`NaiveInvert`](@ref) or [`Jeckelmann`](@ref)"
    flavour::F = NaiveInvert()
    "algorithm used for the linear solvers"
    solver::S = Defaults.linearsolver
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol * 10
    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter
    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity
end

"""
    propagator(ψ₀::AbstractFiniteMPS, z::Number, H::MPOHamiltonian, alg::DynamicalDMRG; init=copy(ψ₀))

Calculate the propagator ``\\frac{1}{E₀ + z - H}|ψ₀⟩`` using the dynamical DMRG
algorithm.
"""
function propagator end

"""
$(TYPEDEF)

An alternative approach to the dynamical DMRG algorithm, without quadratic terms but with a
less controlled approximation.
This algorithm minimizes the following cost function
```math
⟨ψ|(H - E)|ψ⟩ - ⟨ψ|ψ₀⟩ - ⟨ψ₀|ψ⟩
```
which is equivalent to the original approach if
```math
|ψ₀⟩ = (H - E)|ψ⟩
```

See also [`Jeckelmann`](@ref) for the original approach.
"""
struct NaiveInvert <: DDMRG_Flavour end

function propagator(A::AbstractFiniteMPS, z::Number, H::FiniteMPOHamiltonian,
                    alg::DynamicalDMRG{NaiveInvert}; init=copy(A))
    h_envs = environments(init, H) # environments for h
    mixedenvs = environments(init, A) # environments for <init | A>

    ϵ = 2 * alg.tol
    log = IterLog("DDMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0

            for i in [1:(length(A) - 1); length(A):-1:2]
                tos = ac_proj(i, init, A, mixedenvs)

                H_AC = AC_hamiltonian(i, init, H, init, h_envs)
                AC = init.AC[i]
                AC′, convhist = linsolve(H_AC, -tos, AC, alg.solver, -z, one(z))

                ϵ = max(ϵ, norm(AC′ - AC))
                init.AC[i] = AC′

                convhist.converged == 0 &&
                    @warn "propagator ($i) failed to converge: normres = $(convhist.normres)"
            end

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    return dot(A, init), init
end

"""
$(TYPEDEF)

The original flavour of dynamical DMRG, which minimizes the following (quadratic) cost function:
```math
|| (H - E) |ψ₀⟩ - |ψ⟩ ||
```

See also [`NaiveInvert`](@ref) for a less costly but less accurate alternative.

## References

* [Jeckelmann. Phys. Rev. B 66 (2002)](@cite jeckelmann2002)
"""
struct Jeckelmann <: DDMRG_Flavour end

function propagator(A::AbstractFiniteMPS, z, H::FiniteMPOHamiltonian,
                    alg::DynamicalDMRG{Jeckelmann}; init=copy(A))
    ω = real(z)
    η = imag(z)

    envs1 = environments(init, H) # environments for h
    H2, envs2 = squaredenvs(init, H, envs1) # environments for h^2
    mixedenvs = environments(init, A) # environments for <init | A>

    ϵ = 2 * alg.tol
    log = IterLog("DDMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0

            for i in [1:(length(A) - 1); length(A):-1:2]
                tos = ac_proj(i, init, A, mixedenvs)
                H1_AC = AC_hamiltonian(i, init, H, init, envs1)
                H2_AC = AC_hamiltonian(i, init, H2, init, envs2)
                H_AC = LinearCombination((H1_AC, H2_AC), (-2 * ω, 1))
                AC′, convhist = linsolve(H_AC, -η * tos, init.AC[i], alg.solver, abs2(z), 1)

                ϵ = max(ϵ, norm(AC′ - init.AC[i]))
                init.AC[i] = AC′

                convhist.converged == 0 &&
                    @warn "propagator ($i) failed to converge: normres $(convhist.normres)"
            end

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    a = dot(ac_proj(1, init, A, mixedenvs), init.AC[1])
    cb = leftenv(envs1, 1, A) * TransferMatrix(init.AL, H[1:length(A.AL)], A.AL)
    b = zero(a)
    for i in 1:length(cb)
        b += @plansor cb[i][1 2; 3] * init.C[end][3; 4] *
                      rightenv(envs1, length(A), A)[i][4 2; 5] * conj(A.C[end][1; 5])
    end

    v = b / η - ω / η * a + 1im * a
    return v, init
end

function squaredenvs(state::AbstractFiniteMPS, H::FiniteMPOHamiltonian,
                     envs=environments(state, H))
    H² = conj(H) * H
    L = length(state)

    # impose the correct boundary conditions (important for WindowMPS)
    leftstart = _contract_leftenv²(leftenv(envs, 1, state), leftenv(envs, 1, state))
    rightstart = _contract_rightenv²(rightenv(envs, L, state), rightenv(envs, L, state))

    # to construct the squared caches we will first initialize environments
    # then make all data invalid so it will be recalculated
    envs² = environments(state, H², leftstart, rightstart)
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
