"""
$(TYPEDEF)

Single-site optimization algorithm for excitations on top of MPS groundstates.

# Constructors

    ChepigaAnsatz()
    ChepigaAnsatz(; kwargs...)
    ChepigaAnsatz(alg)

Create a `ChepigaAnsatz` algorithm with the given eigensolver, or by passing the
keyword arguments to [`Arnoldi`](@extref KrylovKit.Arnoldi).

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`excitations`](@ref).

# References

* [Chepiga et al. Phys. Rev. B 96 (2017)](@cite chepiga2017)
"""
struct ChepigaAnsatz{A <: KrylovAlgorithm, B} <: Algorithm
    "algorithm used for the eigenvalue solvers"
    alg::A

    "backend for tensor contractions and index manipulations"
    backend::B
end
ChepigaAnsatz(alg::KrylovAlgorithm) =
    ChepigaAnsatz(alg, Defaults.backend())
function ChepigaAnsatz(;
        backend = Defaults.backend(), kwargs...
    )
    if isempty(kwargs)
        alg = Arnoldi(; krylovdim = 30, tol = 1.0e-10, eager = true)
    else
        alg = Arnoldi(; kwargs...)
    end
    return ChepigaAnsatz(alg, backend)
end

function excitations(
        H, alg::ChepigaAnsatz, ψ::FiniteMPS, envs = environments(ψ, H, ψ);
        sector = leftunit(ψ), num::Int = 1, pos::Int = length(ψ) ÷ 2
    )
    1 ≤ pos ≤ length(ψ) || throw(ArgumentError("invalid position $pos"))
    isunit(sector) || error("not yet implemented for charged excitations")
    allocator = default_allocator(ψ, SerialScheduler())

    # add random offset to kickstart Krylov process:
    AC = ψ.AC[pos]
    AC₀ = add(AC, randn!(similar(AC)), eps(real(scalartype(AC)))^(1 / 4))

    H_eff = AC_hamiltonian(pos, ψ, H, ψ, envs; alg.backend, allocator)
    Es, ACs, info = eigsolve(H_eff, AC₀, num + 1, :SR, alg.alg)
    info.converged < num &&
        @warn "excitation failed to converge: normres = $(info.normres)"

    # discard ground state
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
$(TYPEDEF)

Two-site optimization algorithm for excitations on top of MPS groundstates.

# Constructors

    ChepigaAnsatz2()
    ChepigaAnsatz2(; kwargs...)
    ChepigaAnsatz2(alg, trunc)

Create a `ChepigaAnsatz2` algorithm with the given eigensolver and truncation, or by passing the
keyword arguments to [`Arnoldi`](@extref KrylovKit.Arnoldi).

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`excitations`](@ref).

# References

* [Chepiga et al. Phys. Rev. B 96 (2017)](@cite chepiga2017)
"""
struct ChepigaAnsatz2{A <: KrylovAlgorithm, B} <: Algorithm
    "algorithm used for the eigenvalue solvers, defaults to `Arnoldi(; krylovdim = 30, tol = 1.0e-10, eager = true)`"
    alg::A

    "[truncation strategy](@extref MatrixAlgebraKit.TruncationStrategy) used when splitting the optimized two-site tensor, defaults to `notrunc()`"
    trunc::Any

    "backend for tensor contractions and index manipulations"
    backend::B
end
ChepigaAnsatz2(alg::KrylovAlgorithm, trunc) =
    ChepigaAnsatz2(alg, trunc, Defaults.backend())
function ChepigaAnsatz2(;
        trunc = notrunc(), backend = Defaults.backend(),
        kwargs...
    )
    if isempty(kwargs)
        alg = Arnoldi(; krylovdim = 30, tol = 1.0e-10, eager = true)
    else
        alg = Arnoldi(; kwargs...)
    end
    return ChepigaAnsatz2(alg, trunc, backend)
end

function excitations(
        H, alg::ChepigaAnsatz2, ψ::FiniteMPS, envs = environments(ψ, H, ψ);
        sector = leftunit(ψ), num::Int = 1, pos::Int = length(ψ) ÷ 2
    )
    1 ≤ pos ≤ length(ψ) - 1 || throw(ArgumentError("invalid position $pos"))
    isunit(sector) || error("not yet implemented for charged excitations")
    allocator = default_allocator(ψ, SerialScheduler())

    # add random offset to kickstart Krylov process:
    @plansor AC2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
    AC2₀ = add(AC2, randn!(similar(AC2)), eps(real(scalartype(AC2)))^(1 / 4))

    H_eff = AC2_hamiltonian(pos, ψ, H, ψ, envs; alg.backend, allocator)
    Es, AC2s, info = eigsolve(H_eff, AC2₀, num + 1, :SR, alg.alg)
    info.converged < num &&
        @warn "excitation failed to converge: normres = $(info.normres)"

    # discard ground state
    popfirst!(Es)
    popfirst!(AC2s)

    # map back to finitemps
    ψs = map(AC2s) do ac
        ψ′ = copy(ψ)
        AL, C, AR = svd_trunc!(ac; trunc = alg.trunc)
        normalize!(C)
        ψ′.AC[pos] = (AL, complex(C))
        ψ′.AC[pos + 1] = (complex(C), _transpose_front(AR))
        return ψ′
    end

    return Es, ψs
end
