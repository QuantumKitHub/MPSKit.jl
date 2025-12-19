"""
$(TYPEDEF)

Single site infinite DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct IDMRG{A} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol

    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity

    "algorithm used for gauging the MPS"
    alg_gauge = Defaults.alg_gauge()

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A = Defaults.alg_eigsolve()
end

"""
$(TYPEDEF)

Two-site infinite DMRG algorithm for finding the dominant eigenvector.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct IDMRG2{A, S} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol

    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity

    "algorithm used for gauging the MPS"
    alg_gauge = Defaults.alg_gauge()

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A = Defaults.alg_eigsolve()

    "algorithm used for the singular value decomposition"
    alg_svd::S = Defaults.alg_svd()

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the two-site update"
    trscheme::TruncationStrategy
end


# Internal state of the IDMRG algorithm
struct IDMRGState{S, O, E, T}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::Float64 # TODO: Could be any <:Real
    energy::T
end
function IDMRGState{T}(mps::S, operator::O, envs::E, iter::Int, ϵ::Float64, energy) where {S, O, E, T}
    return IDMRGState{S, O, E, T}(mps, operator, envs, iter, ϵ, T(energy))
end

function find_groundstate(mps, operator, alg::alg_type, envs = environments(mps, operator)) where {alg_type <: Union{<:IDMRG, <:IDMRG2}}
    (length(mps) ≤ 1 && alg isa IDMRG2) && throw(ArgumentError("unit cell should be >= 2"))
    log = alg isa IDMRG ? IterLog("IDMRG") : IterLog("IDMRG2")
    mps = copy(mps)
    iter = 0
    ϵ = calc_galerkin(mps, operator, mps, envs)
    E = zero(TensorOperations.promote_contract(scalartype(mps), scalartype(operator)))

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 begin
            E = expectation_value(mps, operator, envs)
            loginit!(log, ϵ, E)
        end
    end

    state = IDMRGState(mps, operator, envs, iter, ϵ, E)
    it = IterativeSolver(alg, state)

    return LoggingExtras.withlevel(; alg.verbosity) do
        for (mps, envs, ϵ, ΔE) in it
            if ϵ ≤ alg.tol
                @infov 2 logfinish!(log, it.iter, ϵ, ΔE)
                break
            end
            if it.iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, it.iter, ϵ, ΔE)
                break
            end
            @infov 3 logiter!(log, it.iter, ϵ, ΔE)
        end

        alg_gauge = updatetol(alg.alg_gauge, it.state.iter, it.state.ϵ)
        ψ′ = InfiniteMPS(it.state.mps.AR[1:end]; alg_gauge.tol, alg_gauge.maxiter)
        envs = recalculate!(it.state.envs, ψ′, it.state.operator, ψ′)
        return ψ′, envs, it.state.ϵ
    end
end

function Base.iterate(
        it::IterativeSolver{alg_type}, state::IDMRGState{<:Any, <:Any, <:Any, T} = it.state
    ) where {alg_type <: Union{<:IDMRG, <:IDMRG2}, T}
    mps, envs, C_old, E_new = localupdate_step!(it, state)

    # error criterion
    C = mps.C[0]
    space_C_old = _firstspace(C_old)
    space_C = _firstspace(C)
    if space_C != space_C_old
        smallest = infimum(space_C_old, space_C)
        e1 = isometry(space_C_old, smallest)
        e2 = isometry(space_C, smallest)
        ϵ = norm(e2' * C * e2 - e1' * C_old * e1)
    else
        ϵ = norm(C - C_old)
    end

    # New energy
    ΔE = (E_new - state.energy) / 2
    (alg_type <: IDMRG2 && length(mps) == 2) && (ΔE /= 2) # This extra factor gives the correct energy per unit cell. I have no clue why right now.

    # update state
    it.state = IDMRGState{T}(mps, state.operator, envs, state.iter + 1, ϵ, E_new)

    return (mps, envs, ϵ, ΔE), it.state
end

function localupdate_step!(
        it::IterativeSolver{<:IDMRG}, state
    )
    alg_eigsolve = updatetol(it.alg_eigsolve, state.iter, state.ϵ)
    return _localupdate_sweep_idmrg!(state.mps, state.operator, state.envs, alg_eigsolve)
end

function localupdate_step!(
        it::IterativeSolver{<:IDMRG2}, state
    )
    alg_eigsolve = updatetol(it.alg_eigsolve, state.iter, state.ϵ)
    return _localupdate_sweep_idmrg2!(state.mps, state.operator, state.envs, alg_eigsolve, it.trscheme, it.alg_svd)
end

function _localupdate_sweep_idmrg!(ψ, H, envs, alg_eigsolve)
    local E
    C_old = ψ.C[0]
    # left to right sweep
    for pos in 1:length(ψ)
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        _, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)
        if pos == length(ψ)
            # AC needed in next sweep
            ψ.AL[pos], ψ.C[pos] = left_orth(ψ.AC[pos]; positive = true)
        else
            ψ.AL[pos], ψ.C[pos] = left_orth!(ψ.AC[pos]; positive = true)
        end
        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
    end

    # right to left sweep
    for pos in length(ψ):-1:1
        h = AC_hamiltonian(pos, ψ, H, ψ, envs)
        E, ψ.AC[pos] = fixedpoint(h, ψ.AC[pos], :SR, alg_eigsolve)

        ψ.C[pos - 1], temp = right_orth!(_transpose_tail(ψ.AC[pos]; copy = (pos == 1)); positive = true)
        ψ.AR[pos] = _transpose_front(temp)

        transfer_rightenv!(envs, ψ, H, ψ, pos - 1)
    end
    return ψ, envs, C_old, E
end


function _localupdate_sweep_idmrg2!(ψ, H, envs, alg_eigsolve, alg_trscheme, alg_svd)
    # sweep from left to right
    for pos in 1:(length(ψ) - 1)
        ac2 = AC2(ψ, pos; kind = :ACAR)
        h_ac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
        _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

        al, c, ar = svd_trunc!(ac2′; trunc = alg_trscheme, alg = alg_svd)
        normalize!(c)

        ψ.AL[pos] = al
        ψ.C[pos] = complex(c)
        ψ.AR[pos + 1] = _transpose_front(ar)
        ψ.AC[pos + 1] = _transpose_front(c * ar)

        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
        transfer_rightenv!(envs, ψ, H, ψ, pos)
    end

    # update the edge
    ψ.AL[end] = ψ.AC[end] / ψ.C[end]
    ψ.AC[1] = _mul_tail(ψ.AL[1], ψ.C[1])
    ac2 = AC2(ψ, 0; kind = :ALAC)
    h_ac2 = AC2_hamiltonian(0, ψ, H, ψ, envs)
    _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

    al, c, ar = svd_trunc!(ac2′; trunc = alg_trscheme, alg = alg_svd)
    normalize!(c)

    ψ.AL[end] = al
    ψ.C[end] = complex(c)
    ψ.AR[1] = _transpose_front(ar)

    ψ.AC[end] = _mul_tail(al, c)
    ψ.AC[1] = _transpose_front(c * ar)
    ψ.AL[1] = ψ.AC[1] / ψ.C[1]

    C_old = complex(c)

    # update environments
    transfer_leftenv!(envs, ψ, H, ψ, 1)
    transfer_rightenv!(envs, ψ, H, ψ, 0)

    # sweep from right to left
    for pos in (length(ψ) - 1):-1:1
        ac2 = AC2(ψ, pos; kind = :ALAC)
        h_ac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
        _, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)

        al, c, ar = svd_trunc!(ac2′; trunc = alg_trscheme, alg = alg_svd)
        normalize!(c)

        ψ.AL[pos] = al
        ψ.AC[pos] = _mul_tail(al, c)
        ψ.C[pos] = complex(c)
        ψ.AR[pos + 1] = _transpose_front(ar)
        ψ.AC[pos + 1] = _transpose_front(c * ar)

        transfer_leftenv!(envs, ψ, H, ψ, pos + 1)
        transfer_rightenv!(envs, ψ, H, ψ, pos)
    end

    # update the edge
    ψ.AC[end] = _mul_front(ψ.C[end - 1], ψ.AR[end])
    ψ.AR[1] = _transpose_front(ψ.C[end] \ _transpose_tail(ψ.AC[1]))
    ac2 = AC2(ψ, 0; kind = :ACAR)
    h_ac2 = AC2_hamiltonian(0, ψ, H, ψ, envs)
    E, ac2′ = fixedpoint(h_ac2, ac2, :SR, alg_eigsolve)
    al, c, ar = svd_trunc!(ac2′; trunc = alg_trscheme, alg = alg_svd)
    normalize!(c)

    ψ.AL[end] = al
    ψ.C[end] = complex(c)
    ψ.AR[1] = _transpose_front(ar)

    ψ.AR[end] = _transpose_front(ψ.C[end - 1] \ _transpose_tail(al * c))
    ψ.AC[1] = _transpose_front(c * ar)

    transfer_leftenv!(envs, ψ, H, ψ, 1)
    transfer_rightenv!(envs, ψ, H, ψ, 0)
    return ψ, envs, C_old, E
end
