#=
    an excitation tensor has 4 legs (1,2),(3,4)
    the first and the last are virtual, the second is physical, the third is the utility leg
=#
"""
$(TYPEDEF)

Optimization algorithm for quasi-particle excitations on top of MPS groundstates.

## Fields

$(TYPEDFIELDS)

## Constructors
    
    QuasiparticleAnsatz()
    QuasiparticleAnsatz(; kwargs...)
    QuasiparticleAnsatz(alg)

Create a `QuasiparticleAnsatz` algorithm with the given algorithm, or by passing the 
keyword arguments to `Arnoldi`.

## References

- [Haegeman et al. Phys. Rev. Let. 111 (2013)](@cite haegeman2013)
"""
struct QuasiparticleAnsatz{A, E} <: Algorithm
    "algorithm used for the eigenvalue solvers"
    alg::A

    "algorithm used for the quasiparticle environments"
    alg_environments::E
end
function QuasiparticleAnsatz(;
        alg_environments = Defaults.alg_environments(; dynamic_tols = false),
        kwargs...
    )
    alg = Defaults.alg_eigsolve(; dynamic_tols = false, kwargs...)
    return QuasiparticleAnsatz(alg, alg_environments)
end

################################################################################
#                           Infinite Excitations                               #
################################################################################

function excitations(H, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP, lenvs, renvs; num::Int = 1)
    E = effective_excitation_renormalization_energy(H, ϕ₀, lenvs, renvs)
    H_eff = EffectiveExcitationHamiltonian(H, lenvs, renvs, E)
    Es, ϕs, convhist = eigsolve(ϕ₀, num, :SR, alg.alg) do ϕ
        return H_eff(ϕ; alg.alg_environments...)
    end
    convhist.converged < num &&
        @warn "excitation failed to converge: normres = $(convhist.normres)"

    return Es, ϕs
end
function excitations(H, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP, lenvs; num = 1, kwargs...)
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = !istopological(ϕ₀) ? lenvs : environments(ϕ₀.right_gs, H; kwargs...)
    return excitations(H, alg, ϕ₀, lenvs, renvs; num, kwargs...)
end
function excitations(H, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP; num = 1, kwargs...)
    # Infer `lenvs` in function body as it depends on `solver`.
    lenvs = environments(ϕ₀.left_gs, H; kwargs...)
    return excitations(H, alg, ϕ₀, lenvs; num, kwargs...)
end

"""
    excitations(H, algorithm::QuasiparticleAnsatz, momentum::Union{Number, Vector{<:Number}},
                left_ψ::InfiniteMPS, [left_environment],
                [right_ψ::InfiniteMPS], [right_environment];
                kwargs...)

Create and optimise infinite quasiparticle states.

# Arguments
- `H::AbstractMPO`: operator for which to find the excitations
- `algorithm::QuasiparticleAnsatz`: optimization algorithm
- `momentum::Union{Number, Vector{<:Number}}`: momentum or list of momenta
- `left_ψ::InfiniteMPS`: left groundstate
- `[left_environment]`: left groundstate environment
- `[right_ψ::InfiniteMPS]`: right groundstate
- `[right_environment]`: right groundstate environment

# Keywords
- `num::Int`: number of excited states to compute
- `solver`: algorithm for the linear solver of the quasiparticle environments
- `sector=one(sectortype(left_ψ))`: charge of the quasiparticle state
- `parallel=true`: enable multi-threading over different momenta
"""
function excitations(
        H, alg::QuasiparticleAnsatz, momentum::Number, lmps::InfiniteMPS,
        lenvs = environments(lmps, H), rmps::InfiniteMPS = lmps,
        renvs = lmps === rmps ? lenvs : environments(rmps, H);
        sector = one(sectortype(lmps)), kwargs...
    )
    ϕ₀ = LeftGaugedQP(rand, lmps, rmps; sector, momentum)
    return excitations(H, alg, ϕ₀, lenvs, renvs; kwargs...)
end
function excitations(
        H, alg::QuasiparticleAnsatz, momenta, lmps,
        lenvs = environments(lmps, H), rmps = lmps,
        renvs = lmps === rmps ? lenvs : environments(rmps, H);
        verbosity = Defaults.verbosity, num = 1,
        sector = one(sectortype(lmps)), parallel = true, kwargs...
    )
    # wrapper to evaluate sector as positional argument
    Toutput = let
        function wrapper(H, alg, p, lmps, lenvs, rmps, renvs, sector; kwargs...)
            return excitations(
                H, alg, p, lmps, lenvs, rmps, renvs;
                sector, kwargs...
            )
        end
        Core.Compiler.return_type(
            wrapper,
            Tuple{
                typeof(H), typeof(alg),
                eltype(momenta), typeof(lmps),
                typeof(lenvs),
                typeof(rmps), typeof(renvs), typeof(sector),
            }
        )
    end

    results = similar(momenta, Toutput)
    scheduler = parallel ? :greedy : :serial
    tmap!(results, momenta; scheduler) do momentum
        E, ϕ = excitations(
            H, alg, momentum, lmps, lenvs, rmps, renvs; num, kwargs...,
            sector
        )
        verbosity ≥ VERBOSE_CONV && @info "Found excitations for momentum = $(momentum)"
        return E, ϕ
    end

    Ep = permutedims(reduce(hcat, map(x -> x[1][1:num], results)))
    Bp = permutedims(reduce(hcat, map(x -> x[2][1:num], results)))

    return Ep, Bp
end

################################################################################
#                           Finite Excitations                                 #
################################################################################

function excitations(
        H, alg::QuasiparticleAnsatz, ϕ₀::FiniteQP,
        lenvs = environments(ϕ₀.left_gs, H),
        renvs = !istopological(ϕ₀) ? lenvs : environments(ϕ₀.right_gs, H);
        num = 1
    )
    E = effective_excitation_renormalization_energy(H, ϕ₀, lenvs, renvs)
    H_eff = EffectiveExcitationHamiltonian(H, lenvs, renvs, E)
    Es, ϕs, convhist = eigsolve(ϕ₀, num, :SR, alg.alg) do ϕ
        return H_eff(ϕ; alg.alg_environments...)
    end

    convhist.converged < num &&
        @warn "excitation failed to converge: normres = $(convhist.normres)"

    return Es, ϕs
end

"""
    excitations(H, algorithm::QuasiparticleAnsatz, left_ψ::InfiniteMPS, [left_environment],
                [right_ψ::InfiniteMPS], [right_environment]; kwargs...)

Create and optimise finite quasiparticle states.

# Arguments
- `H::AbstractMPO`: operator for which to find the excitations
- `algorithm::QuasiparticleAnsatz`: optimization algorithm
- `left_ψ::FiniteMPS`: left groundstate
- `[left_environment]`: left groundstate environment
- `[right_ψ::FiniteMPS]`: right groundstate
- `[right_environment]`: right groundstate environment

# Keywords
- `num::Int`: number of excited states to compute
- `sector=one(sectortype(left_ψ))`: charge of the quasiparticle state
"""
function excitations(
        H, alg::QuasiparticleAnsatz, lmps::FiniteMPS,
        lenvs = environments(lmps, H), rmps::FiniteMPS = lmps,
        renvs = lmps === rmps ? lenvs : environments(rmps, H);
        sector = one(sectortype(lmps)), num = 1
    )
    ϕ₀ = LeftGaugedQP(rand, lmps, rmps; sector)
    return excitations(H, alg, ϕ₀, lenvs, renvs; num)
end

################################################################################
#                           Statmech Excitations                               #
################################################################################

function excitations(
        H::MultilineMPO, alg::QuasiparticleAnsatz, ϕ₀::MultilineQP, lenvs, renvs;
        num = 1
    )
    H_effs = map(parent(H), parent(ϕ₀), parent(lenvs), parent(renvs)) do h, ϕ, lenv, renv
        E₀ = effective_excitation_renormalization_energy(h, ϕ, lenv, renv)
        return EffectiveExcitationHamiltonian(h, lenv, renv, E₀)
    end
    H_eff = Multiline(H_effs)

    Es, ϕs, convhist = eigsolve(ϕ₀, num, :LM, alg.alg) do ϕ
        return H_eff(ϕ; alg.alg_environments...)
    end
    convhist.converged < num &&
        @warn "excitation failed to converge: normres = $(convhist.normres)"

    return Es, ϕs
end

function excitations(
        H::InfiniteMPO, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP, lenvs, renvs;
        num = 1
    )
    E = effective_excitation_renormalization_energy(H, ϕ₀, lenvs, renvs)
    H_eff = EffectiveExcitationHamiltonian(H_eff, lenvs, renvs, E)

    Es, ϕs, convhist = eigsolve(ϕ₀, num, :LM, alg.alg) do ϕ
        return H_eff(ϕ; alg.alg_environments...)
    end
    convhist.converged < num &&
        @warn "excitation failed to converge: normres = $(convhist.normres)"

    return Es, ϕs
end

function excitations(
        H::MultilineMPO, alg::QuasiparticleAnsatz, ϕ₀::MultilineQP, lenvs;
        kwargs...
    )
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = !istopological(ϕ₀) ? lenvs : environments(ϕ₀.right_gs, H; kwargs...)
    return excitations(H, alg, ϕ₀, lenvs, renvs; kwargs...)
end
function excitations(
        H::MultilineMPO, alg::QuasiparticleAnsatz, ϕ₀::MultilineQP;
        num = 1, kwargs...
    )
    # Infer `lenvs` in function body as it depends on `solver`.
    lenvs = environments(ϕ₀.left_gs, H; kwargs...)
    return excitations(H, alg, ϕ₀, lenvs; num, kwargs...)
end

function excitations(
        H::MPO, alg::QuasiparticleAnsatz, momentum::Real, lmps::InfiniteMPS,
        lenvs = environments(lmps, H), rmps::InfiniteMPS = lmps,
        renvs = lmps === rmps ? lenvs : environments(rmps, H);
        kwargs...
    )
    multiline_H = convert(MultilineMPO, H)
    multiline_lmps = convert(MultilineMPS, lmps)
    lenvs′ = Multiline([lenvs])
    if lmps === rmps
        multiline_rmps = multiline_lmps
        renvs′ = lenvs′
    else
        multiline_rmps = convert(MultilineMPS, rmps)
        renvs′ = Multiline([renvs])
    end

    return excitations(
        multiline_H, alg, momentum, multiline_lmps, lenvs′, multiline_rmps, renvs′;
        kwargs...
    )
end

function excitations(
        H::MultilineMPO, alg::QuasiparticleAnsatz, momentum::Real, lmps::MultilineMPS,
        lenvs = environments(lmps, H), rmps = lmps,
        renvs = lmps === rmps ? lenvs : environments(rmps, H);
        sector = one(sectortype(lmps)), kwargs...
    )
    ϕ₀ = LeftGaugedQP(randn, lmps, rmps; sector, momentum)
    return excitations(H, alg, ϕ₀, lenvs, renvs; kwargs...)
end

################################################################################
#                                H_eff                                         #
################################################################################

struct EffectiveExcitationHamiltonian{TO, TGL, TGR, E}
    operator::TO
    lenvs::TGL
    renvs::TGR
    energy::E
end
# to allow Multiline checks
Base.length(H::EffectiveExcitationHamiltonian) = length(H.operator)

function (H::EffectiveExcitationHamiltonian)(ϕ::QP; kwargs...)
    qp_envs = environments(ϕ, H.operator, H.lenvs, H.renvs; kwargs...)
    return effective_excitation_hamiltonian(H.operator, ϕ, qp_envs, H.energy)
end
function (H::Multiline{<:EffectiveExcitationHamiltonian})(ϕ::MultilineQP; kwargs...)
    return Multiline(map((x, y) -> x(y; kwargs...), parent(H), parent(ϕ)))
end

function effective_excitation_hamiltonian(H, ϕ, envs = environments(ϕ, H))
    E₀ = effective_excitation_renormalization_energy(H, ϕ, envs.leftenvs, envs.rightenvs)
    return effective_excitation_hamiltonian(H, ϕ, envs, E₀)
end
function effective_excitation_hamiltonian(H, ϕ, qp_envs, E)
    ϕ′ = similar(ϕ)
    tforeach(1:length(ϕ); scheduler = Defaults.scheduler[]) do loc
        ϕ′[loc] = _effective_excitation_local_apply(loc, ϕ, H, E[loc], qp_envs)
        return nothing
    end
    return ϕ′
end

function effective_excitation_hamiltonian(
        H::MultilineMPO, ϕ::MultilineQP, envs = environments(ϕ, H)
    )
    E₀ = map(
        effective_excitation_renormalization_energy, parent(H), parent(ϕ),
        parent(envs).leftenvs, parent(envs).rightenvs
    )
    return effective_excitation_hamiltonian(H, ϕ, envs, E₀)
end
function effective_excitation_hamiltonian(H::MultilineMPO, ϕ::MultilineQP, envs, E)
    return Multiline(
        map(effective_excitation_hamiltonian, parent(H), parent(ϕ), parent(envs), E)
    )
end

function _effective_excitation_local_apply(site, ϕ, H::MPOHamiltonian, E::Number, envs)
    B = ϕ[site]
    GL = leftenv(envs.leftenvs, site, ϕ.left_gs)
    GR = rightenv(envs.rightenvs, site, ϕ.right_gs)

    # renormalize first -> allocates destination
    B′ = scale(B, -E)

    # B in center
    @plansor B′[-1 -2; -3 -4] += GL[-1 5; 4] * B[4 2; -3 1] * H[site][5 -2; 2 3] * GR[1 3; -4]

    # B to the left
    if site > 1 || ϕ isa InfiniteQP
        AR = ϕ.right_gs.AR[site]
        GBL = envs.leftBenvs[site]
        @plansor B′[-1 -2; -3 -4] += GBL[-1 4; -3 5] * AR[5 2; 1] * H[site][4 -2; 2 3] * GR[1 3; -4]
    end

    # B to the right
    if site < length(ϕ.left_gs) || ϕ isa InfiniteQP
        AL = ϕ.left_gs.AL[site]
        GBR = envs.rightBenvs[site]
        @plansor B′[-1 -2; -3 -4] += GL[-1 2; 1] * AL[1 3; 4] * H[site][2 -2; 3 5] * GBR[4 5; -3 -4]
    end

    return B′
end

function _effective_excitation_local_apply(site, ϕ, H::MPO, E::Number, envs)
    left_gs = ϕ.left_gs
    right_gs = ϕ.right_gs

    B = ϕ[site]
    GL = leftenv(envs.leftenvs, site, ϕ.left_gs)
    GR = rightenv(envs.rightenvs, site, ϕ.right_gs)

    @plansor T[-1 -2; -3 -4] := GL[-1 5; 4] * B[4 2; -3 1] * H[site][5 -2; 2 3] * GR[1 3; -4]

    @plansor T[-1 -2; -3 -4] += envs.leftBenvs[site][-1 4; -3 5] *
        right_gs.AR[site][5 2; 1] * H[site][4 -2; 2 3] * GR[1 3; -4]

    @plansor T[-1 -2; -3 -4] += GL[-1 2; 1] * left_gs.AL[site][1 3; 4] *
        H[site][2 -2; 3 5] * envs.rightBenvs[site][4 5; -3 -4]

    return scale!(T, inv(E))
end

function effective_excitation_renormalization_energy(H, ϕ, lenvs, renvs)
    ψ_left = ϕ.left_gs
    ψ_right = ϕ.right_gs
    E = Vector{scalartype(ϕ)}(undef, length(ϕ))
    for i in eachindex(E)
        E[i] = contract_mpo_expval(
            ψ_left.AC[i], leftenv(lenvs, i, ψ_left), H[i], rightenv(lenvs, i, ψ_left)
        )
        if istopological(ϕ)
            E[i] += contract_mpo_expval(
                ψ_right.AC[i], leftenv(renvs, i, ψ_right), H[i], rightenv(renvs, i, ψ_right)
            )
            E[i] /= 2
        end
    end
    return E
end
