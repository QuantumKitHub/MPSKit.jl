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
struct QuasiparticleAnsatz{A} <: Algorithm
    "algorithm used for the eigenvalue solvers"
    alg::A
end
function QuasiparticleAnsatz(; kwargs...)
    alg = Defaults.alg_eigsolve(; dynamic_tols=false, kwargs...)
    return QuasiparticleAnsatz(alg)
end

################################################################################
#                           Infinite Excitations                               #
################################################################################

function excitations(H, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP, lenvs, renvs;
                     num=1, kwargs...)
    qp_envs(ϕ) = environments(ϕ, H, lenvs, renvs; kwargs...)
    E = effective_excitation_renormalization_energy(H, ϕ₀, lenvs, renvs)
    H_eff(ϕ) = effective_excitation_hamiltonian(H, ϕ, qp_envs(ϕ), E)

    Es, ϕs, convhist = eigsolve(H_eff, ϕ₀, num, :SR, alg.alg)
    convhist.converged < num &&
        @warn "excitation failed to converge: normres = $(convhist.normres)"

    return Es, ϕs
end
function excitations(H, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP, lenvs;
                     num=1, kwargs...)
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = ϕ₀.trivial ? lenvs : environments(ϕ₀.right_gs, H; kwargs...)
    return excitations(H, alg, ϕ₀, lenvs, renvs; num, kwargs...)
end
function excitations(H, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP;
                     num=1, kwargs...)
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
function excitations(H, alg::QuasiparticleAnsatz, momentum::Number, lmps::InfiniteMPS,
                     lenvs=environments(lmps, H), rmps::InfiniteMPS=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     sector=one(sectortype(lmps)), num=1, kwargs...)
    ϕ₀ = LeftGaugedQP(rand, lmps, rmps; sector, momentum)
    return excitations(H, alg, ϕ₀, lenvs, renvs; num, kwargs...)
end
function excitations(H, alg::QuasiparticleAnsatz, momenta, lmps,
                     lenvs=environments(lmps, H), rmps=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     verbosity=Defaults.verbosity, num=1,
                     sector=one(sectortype(lmps)), parallel=true, kwargs...)
    if parallel
        tasks = map(momenta) do momentum
            Threads.@spawn begin
                E, ϕ = excitations(H, alg, momentum, lmps, lenvs, rmps, renvs; num,
                                   kwargs...,
                                   sector)
                verbosity ≥ VERBOSE_CONV &&
                    @info "Found excitations for momentum = $(momentum)"
                return E, ϕ
            end
        end

        fetched = fetch.(tasks)
    else
        fetched = map(momenta) do momentum
            E, ϕ = excitations(H, alg, momentum, lmps, lenvs, rmps, renvs; num, kwargs...,
                               sector)
            verbosity ≥ VERBOSE_CONV && @info "Found excitations for momentum = $(momentum)"
            return E, ϕ
        end
    end

    Ep = permutedims(reduce(hcat, map(x -> x[1][1:num], fetched)))
    Bp = permutedims(reduce(hcat, map(x -> x[2][1:num], fetched)))

    return Ep, Bp
end

################################################################################
#                           Finite Excitations                                 #
################################################################################

function excitations(H, alg::QuasiparticleAnsatz, ϕ₀::FiniteQP,
                     lenvs=environments(ϕ₀.left_gs, H),
                     renvs=ϕ₀.trivial ? lenvs : environments(ϕ₀.right_gs, H); num=1)
    qp_envs(ϕ) = environments(ϕ, H, lenvs, renvs)
    E = effective_excitation_renormalization_energy(H, ϕ₀, lenvs, renvs)
    H_eff(ϕ) = effective_excitation_hamiltonian(H, ϕ, qp_envs(ϕ), E)
    Es, ϕs, convhist = eigsolve(H_eff, ϕ₀, num, :SR, alg.alg)

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
function excitations(H, alg::QuasiparticleAnsatz, lmps::FiniteMPS,
                     lenvs=environments(lmps, H), rmps::FiniteMPS=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     sector=one(sectortype(lmps)), num=1)
    ϕ₀ = LeftGaugedQP(rand, lmps, rmps; sector)
    return excitations(H, alg, ϕ₀, lenvs, renvs; num)
end

################################################################################
#                           Statmech Excitations                               #
################################################################################

function excitations(H::MultilineMPO, alg::QuasiparticleAnsatz, ϕ₀::MultilineQP,
                     lenvs, renvs; num=1, kwargs...)
    qp_envs(ϕ) = environments(ϕ, H, lenvs, renvs; kwargs...)
    function H_eff(ϕ′)
        ϕ = Multiline(ϕ′)
        return effective_excitation_hamiltonian(H, ϕ, qp_envs(ϕ)).data.data
    end

    Es, ϕs, convhist = eigsolve(H_eff, ϕ₀.data.data, num, :LM, alg.alg)
    convhist.converged < num &&
        @warn "excitation failed to converge: normres = $(convhist.normres)"

    return Es, map(Multiline, ϕs)
end

function excitations(H::InfiniteMPO, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP, lenvs, renvs;
                     num=1, kwargs...)
    qp_envs(ϕ) = environments(ϕ, H, lenvs, renvs; kwargs...)
    H_eff(ϕ) = effective_excitation_hamiltonian(H, ϕ, qp_envs(ϕ))

    Es, ϕs, convhist = eigsolve(H_eff, ϕ₀, num, :LM, alg.alg)
    convhist.converged < num &&
        @warn "excitation failed to converge: normres = $(convhist.normres)"

    return Es, ϕs
end

function excitations(H::MultilineMPO, alg::QuasiparticleAnsatz, ϕ₀::MultilineQP,
                     lenvs; num=1, kwargs...)
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = ϕ₀.trivial ? lenvs : environments(ϕ₀.right_gs, H; kwargs...)
    return excitations(H, alg, ϕ₀, lenvs, renvs; num, kwargs...)
end
function excitations(H::MultilineMPO, alg::QuasiparticleAnsatz, ϕ₀::MultilineQP;
                     num=1, kwargs...)
    # Infer `lenvs` in function body as it depends on `solver`.
    lenvs = environments(ϕ₀.left_gs, H; solver)
    return excitations(H, alg, ϕ₀, lenvs; num, solver)
end

function excitations(H::DenseMPO, alg::QuasiparticleAnsatz, momentum::Real,
                     lmps::InfiniteMPS,
                     lenvs=environments(lmps, H), rmps::InfiniteMPS=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     sector=one(sectortype(lmps)), num=1, kwargs...)
    multiline_lmps = convert(MultilineMPS, lmps)
    lenvs′ = Multiline([lenvs])
    if lmps === rmps
        excitations(convert(MultilineMPO, H), alg, momentum, multiline_lmps, lenvs′,
                    multiline_lmps,
                    lenvs′; sector, num, kwargs...)
    else
        renvs′ = Multiline([renvs])
        excitations(convert(MultilineMPO, H), alg, momentum, multiline_lmps, lenvs′,
                    convert(MultilineMPS, rmps), renvs′; sector, num, kwargs...)
    end
end

function excitations(H::MultilineMPO, alg::QuasiparticleAnsatz, momentum::Real,
                     lmps::MultilineMPS,
                     lenvs=environments(lmps, H), rmps=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     sector=one(sectortype(lmps)), num=1, kwargs...)
    ϕ₀ = Multiline(map(1:size(lmps, 1)) do row
                       return LeftGaugedQP(rand, lmps[row], rmps[row]; sector, momentum)
                   end)

    return excitations(H, alg, ϕ₀, lenvs, renvs; num, kwargs...)
end

################################################################################
#                                H_eff                                         #
################################################################################

function effective_excitation_hamiltonian(H::Union{InfiniteMPOHamiltonian,
                                                   FiniteMPOHamiltonian}, ϕ::QP,
                                          envs=environments(ϕ, H),
                                          energy=effective_excitation_renormalization_energy(H,
                                                                                             ϕ,
                                                                                             envs.leftenvs,
                                                                                             envs.rightenvs))
    ϕ′ = similar(ϕ)
    tforeach(1:length(ϕ); scheduler=Defaults.scheduler[]) do loc
        return ϕ′[loc] = _effective_excitation_local_apply(loc, ϕ, H, energy[loc], envs)
    end
    return ϕ′
end

function effective_excitation_hamiltonian(H::MultilineMPO, ϕ::MultilineQP,
                                          envs=environments(ϕ, H))
    return Multiline(map(effective_excitation_hamiltonian,
                         parent(H), parent(ϕ), parent(envs)))
end

function effective_excitation_hamiltonian(H::InfiniteMPO, ϕ::InfiniteQP,
                                          envs=environments(ϕ, H))
    ϕ′ = similar(ϕ)
    left_gs = ϕ.left_gs
    right_gs = ϕ.right_gs

    Bs = [ϕ[i] for i in 1:length(H)]
    for site in 1:length(ϕ)
        en = @plansor conj(left_gs.AC[site][2 6; 4]) *
                      leftenv(envs.leftenvs, site, left_gs)[2 5; 3] *
                      left_gs.AC[site][3 7; 1] *
                      H[site][5 6; 7 8] *
                      rightenv(envs.leftenvs, site, left_gs)[1 8; 4]

        @plansor T[-1 -2; -3 -4] := leftenv(envs.leftenvs, site, left_gs)[-1 5; 4] *
                                    Bs[site][4 2; -3 1] *
                                    H[site][5 -2; 2 3] *
                                    rightenv(envs.rightenvs, site, right_gs)[1 3; -4]

        @plansor T[-1 -2; -3 -4] += envs.leftBenvs[site][-1 4; -3 5] *
                                    right_gs.AR[site][5 2; 1] *
                                    H[site][4 -2; 2 3] *
                                    rightenv(envs.rightenvs, site, right_gs)[1 3; -4]

        @plansor T[-1 -2; -3 -4] += leftenv(envs.leftenvs, site, left_gs)[-1 2; 1] *
                                    left_gs.AL[site][1 3; 4] *
                                    H[site][2 -2; 3 5] *
                                    envs.rightBenvs[site][4 5; -3 -4]

        ϕ′[site] = T / en
    end

    return ϕ′
end

function _effective_excitation_local_apply(site, ϕ,
                                           H::Union{InfiniteMPOHamiltonian,
                                                    FiniteMPOHamiltonian}, E::Number,
                                           envs)
    B = ϕ[site]
    GL = leftenv(envs.leftenvs, site, ϕ.left_gs)
    GR = rightenv(envs.rightenvs, site, ϕ.right_gs)

    # renormalize first -> allocates destination
    B′ = scale(B, -E)

    # B in center
    @plansor B′[-1 -2; -3 -4] += GL[-1 5; 4] *
                                 B[4 2; -3 1] *
                                 H[site][5 -2; 2 3] *
                                 GR[1 3; -4]

    # B to the left
    if site > 1 || ϕ isa InfiniteQP
        AR = ϕ.right_gs.AR[site]
        GBL = envs.leftBenvs[site]
        @plansor B′[-1 -2; -3 -4] += GBL[-1 4; -3 5] *
                                     AR[5 2; 1] *
                                     H[site][4 -2; 2 3] *
                                     GR[1 3; -4]
    end

    # B to the right
    if site < length(ϕ.left_gs) || ϕ isa InfiniteQP
        AL = ϕ.left_gs.AL[site]
        GBR = envs.rightBenvs[site]
        @plansor B′[-1 -2; -3 -4] += GL[-1 2; 1] *
                                     AL[1 3; 4] *
                                     H[site][2 -2; 3 5] *
                                     GBR[4 5; -3 -4]
    end

    return B′
end

function effective_excitation_renormalization_energy(H, ϕ, lenvs, renvs)
    ψ_left = ϕ.left_gs
    ψ_right = ϕ.right_gs
    E = Vector{scalartype(ϕ)}(undef, length(ϕ))
    for i in eachindex(E)
        E[i] = contract_mpo_expval(ψ_left.AC[i], leftenv(lenvs, i, ψ_left),
                                   H[i], rightenv(lenvs, i, ψ_left))
        if !ϕ.trivial
            E[i] += contract_mpo_expval(ψ_right.AC[i], leftenv(renvs, i, ψ_right),
                                        H[i], rightenv(renvs, i, ψ_right))
            E[i] /= 2
        end
    end
    return E
end
