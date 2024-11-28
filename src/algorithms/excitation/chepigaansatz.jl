
"""
    ChepigaAnsatz <: Algorithm

Optimization algorithm for excitations on top of MPS groundstates, as
introduced in this [paper](https://doi.org/10.1103/PhysRevB.96.054425).

## Fields
- `alg::A = Defaults.eigsolver`: algorithm to use for the eigenvalue problem.

## Constructors

    ChepigaAnsatz()
    ChepigaAnsatz(; kwargs...)
    ChepigaAnsatz(alg)

Create a `ChepigaAnsatz` algorithm with the given eigensolver, or by passing the
keyword arguments to `Arnoldi`.
"""
struct ChepigaAnsatz{A} <: Algorithm
    alg::A
end
function ChepigaAnsatz(; kwargs...)
    if isempty(kwargs)
        alg = Arnoldi(; krylovdim=30, tol=1e-10, eager=true)
    else
        alg = Arnoldi(; kwargs...)
    end
    return ChepigaAnsatz(alg)
end

function excitations(H, alg::ChepigaAnsatz, ψ::FiniteMPS, envs=environments(ψ, H);
                     sector=one(sectortype(ψ)), num::Int=1, pos::Int=length(ψ) ÷ 2)
    1 ≤ pos ≤ length(ψ) || throw(ArgumentError("invalid position $pos"))
    sector == one(sector) || error("not yet implemented for charged excitations")

    # add random offset to kickstart Krylov process:
    AC = ψ.AC[pos]
    AC₀ = add(AC, TensorMap(randn, scalartype(AC), space(AC)),
              eps(real(scalartype(AC)))^(1 / 4))

    H_eff = ∂∂AC(pos, ψ, H, envs)
    Es, ACs, info = eigsolve(H_eff, AC₀, num + 1, :SR, alg.alg)
    info.converged < num &&
        @warn "excitation failed to converge: normres = $(info.normres)"

    # discard groundstate
    popfirst!(Es)
    popfirst!(ACs)

    # map back to finitemps
    ψs = map(ACs) do ac
        ψ′ = copy(ψ)
        ψ′.AC[pos] = ac
        return ψ′
    end

    return Es, ψs
end

"""
    ChepigaAnsatz2 <: Algorithm

Optimization algorithm for excitations on top of MPS groundstates, as
introduced in this [paper](https://doi.org/10.1103/PhysRevB.96.054425).

## Fields
- `alg::A = Defaults.eigsolver`: algorithm to use for the eigenvalue problem.
- `trscheme = Defaults.trscheme`: algorithm to use for truncation.

## Constructors

    ChepigaAnsatz2()
    ChepigaAnsatz2(; kwargs...)
    ChepigaAnsatz2(alg, trscheme)

Create a `ChepigaAnsatz2` algorithm with the given eigensolver and truncation, or by passing the
keyword arguments to `Arnoldi`.
"""
struct ChepigaAnsatz2{A} <: Algorithm
    alg::A
    trscheme::Any
end
function ChepigaAnsatz2(; trscheme=notrunc(), kwargs...)
    if isempty(kwargs)
        alg = Arnoldi(; krylovdim=30, tol=1e-10, eager=true)
    else
        alg = Arnoldi(; kwargs...)
    end
    return ChepigaAnsatz2(alg, trscheme)
end

function excitations(H, alg::ChepigaAnsatz2, ψ::FiniteMPS, envs=environments(ψ, H);
                     sector=one(sectortype(ψ)), num::Int=1, pos::Int=length(ψ) ÷ 2)
    1 ≤ pos ≤ length(ψ) - 1 || throw(ArgumentError("invalid position $pos"))
    sector == one(sector) || error("not yet implemented for charged excitations")

    # add random offset to kickstart Krylov process:
    @plansor AC2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
    AC2₀ = add(AC2, TensorMap(randn, scalartype(AC2), space(AC2)),
               eps(real(scalartype(AC2)))^(1 / 4))

    H_eff = ∂∂AC2(pos, ψ, H, envs)
    Es, AC2s, info = eigsolve(H_eff, AC2₀, num + 1, :SR, alg.alg)
    info.converged < num &&
        @warn "excitation failed to converge: normres = $(info.normres)"

    # discard groundstate
    popfirst!(Es)
    popfirst!(AC2s)

    # map back to finitemps
    ψs = map(AC2s) do ac
        ψ′ = copy(ψ)
        AL, C, AR, = tsvd!(ac; trunc=alg.trscheme)
        normalize!(C)
        ψ′.AC[pos] = (AL, complex(C))
        ψ′.AC[pos + 1] = (complex(C), _transpose_front(AR))
        return ψ′
    end

    return Es, ψs
end
