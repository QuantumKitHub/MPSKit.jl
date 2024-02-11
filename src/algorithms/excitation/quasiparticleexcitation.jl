#=
    an excitation tensor has 4 legs (1,2),(3,4)
    the first and the last are virtual, the second is physical, the third is the utility leg
=#
"""
    QuasiparticleAnsatz <: Algorithm

Optimization algorithm for quasiparticle excitations on top of MPS groundstates, as
introduced in this [paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.080401).

# Fields
- `toler::Float64`: tolerance for convergence criterium
- `krylovdim::Int`: Krylov subspace dimension
"""
@kwdef struct QuasiparticleAnsatz <: Algorithm
    toler::Float64 = 1e-10
    krylovdim::Int = 30
end

################################################################################
#                           Infinite Excitations                               #
################################################################################

function excitations(H, alg::QuasiparticleAnsatz, ϕ₀::InfiniteQP, lenvs, renvs;
                     num=1, solver=Defaults.linearsolver)
    qp_envs(ϕ) = enϕironments(ϕ, H, lenvs, renvs; solver=solver)
    E = effective_excitation_renormalization_energy(H, ϕ₀, lenvs, renvs)
    H_eff = @closure(ϕ -> effective_excitation_hamiltonian(H, ϕ, qp_envs(ϕ), E))

    Es, Vs, convhist = eigsolve(H_eff, ϕ₀, num, :SR; tol=alg.toler, krylovdim=alg.krylovdim)
    convhist.converged < num && @warn "Quasiparticle didn't converge: $(convhist.normres)"

    return Es, Vs
end
function excitations(H, alg::QuasiparticleAnsatz, V₀::InfiniteQP, lenvs;
                     num=1, solver=Defaults.linearsolver)
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = V₀.trivial ? lenvs : environments(V₀.right_gs, H; solver=solver)
    return excitations(H, alg, V₀, lenvs, renvs; num, solver)
end
function excitations(H, alg::QuasiparticleAnsatz, V₀::InfiniteQP;
                     num=1, solver=Defaults.linearsolver)
    # Infer `lenvs` in function body as it depends on `solver`.
    lenvs = environments(V₀.left_gs, H; solver=solver)
    return excitations(H, alg, V₀, lenvs; num, solver)
end

"""
    excitations(H, algorithm::QuasiparticleAnsatz, p::Union{Number, Vector{<:Number}},
                left_ψ::InfiniteMPS, [left_environment],
                [right_ψ::InfiniteMPS], [right_environment];
                kwargs...)

Create and optimise infinite quasiparticle states.

# Arguments
- `H::AbstractMPO`: operator for which to find the excitations
- `algorithm::QuasiparticleAnsatz`: optimization algorithm
- `p::Union{Number, Vector{<:Number}}`: momentum or list of momenta
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
function excitations(H, alg::QuasiparticleAnsatz, p::Number, lmps::InfiniteMPS,
                     lenvs=environments(lmps, H), rmps::InfiniteMPS=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     sector=one(sectortype(lmps)), num=1, solver=Defaults.linearsolver)
    V₀ = LeftGaugedQP(rand, lmps, rmps; sector=sector, momentum=p)
    return excitations(H, alg, V₀, lenvs, renvs; num=num, solver=solver)
end
function excitations(H, alg::QuasiparticleAnsatz, momenta, lmps,
                     lenvs=environments(lmps, H), rmps=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     verbose=Defaults.verbose, num=1, solver=Defaults.linearsolver,
                     sector=one(sectortype(lmps)), parallel=true)
    if parallel
        tasks = map(momenta) do p
            Threads.@spawn begin
                E, V = excitations(H, alg, p, lmps, lenvs, rmps, renvs; num, solver, sector)
                verbose && @info "Found excitations for p = $(p)"
                return E, V
            end
        end

        fetched = fetch.(tasks)
    else
        fetched = map(momenta) do p
            E, V = excitations(H, alg, p, lmps, lenvs, rmps, renvs; num, solver, sector)
            verbose && @info "Found excitations for p = $(p)"
            return E, V
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
    qp_envs(V) = environments(V, H, lenvs, renvs)
    E = effective_excitation_renormalization_energy(H, ϕ₀, lenvs, renvs)
    H_eff = @closure(V -> effective_excitation_hamiltonian(H, V, qp_envs(V), E))

    Es, Vs, convhist = eigsolve(H_eff, ϕ₀, num, :SR; tol=alg.toler, krylovdim=alg.krylovdim)

    convhist.converged < num && @warn "Quasiparticle didn't converge: $(convhist.normres)"

    return Es, Vs
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
    V₀ = LeftGaugedQP(rand, lmps, rmps; sector=sector)
    return excitations(H, alg, V₀, lenvs, renvs; num=num)
end

################################################################################
#                           Statmech Excitations                               #
################################################################################

function excitations(H::MPOMultiline, alg::QuasiparticleAnsatz, V₀::Multiline{<:InfiniteQP},
                     lenvs, renvs; num=1, solver=Defaults.linearsolver)
    qp_envs(V) = environments(V, H, lenvs, renvs; solver=solver)
    function H_eff(oV)
        V = Multiline(oV.vecs)
        return RecursiveVec(effective_excitation_hamiltonian(H, V, qp_envs(V)).data.data)
    end

    Es, Vs, convhist = eigsolve(H_eff, RecursiveVec(V₀.data.data), num, :LM;
                                tol=alg.toler, krylovdim=alg.krylovdim)

    convhist.converged < num && @warn "Quasiparticle didn't converge: $(convhist.normres)"

    return Es, map(x -> Multiline(x.vecs), Vs)
end

function excitations(H::MPOMultiline, alg::QuasiparticleAnsatz, V₀::Multiline{<:InfiniteQP},
                     lenvs; num=1, solver=Defaults.linearsolver)
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = V₀.trivial ? lenvs : environments(V₀.right_gs, H; solver=solver)
    return excitations(H, alg, V₀, lenvs, renvs; num, solver)
end
function excitations(H::MPOMultiline, alg::QuasiparticleAnsatz, V₀::Multiline{<:InfiniteQP};
                     num=1, solver=Defaults.linearsolver)
    # Infer `lenvs` in function body as it depends on `solver`.
    lenvs = environments(V₀.left_gs, H; solver=solver)
    return excitations(H, alg, V₀, lenvs; num, solver)
end

function excitations(H::DenseMPO, alg::QuasiparticleAnsatz, p::Real, lmps::InfiniteMPS,
                     lenvs=environments(lmps, H), rmps::InfiniteMPS=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     sector=one(sectortype(lmps)), num=1, solver=Defaults.linearsolver)
    multiline_lmps = convert(MPSMultiline, lmps)
    if lmps === rmps
        excitations(convert(MPOMultiline, H), alg, p, multiline_lmps, lenvs, multiline_lmps,
                    lenvs; sector, num, solver)
    else
        excitations(convert(MPOMultiline, H), alg, p, multiline_lmps, lenvs,
                    convert(MPSMultiline, rmps), renvs; sector, num, solver)
    end
end

function excitations(H::MPOMultiline, alg::QuasiparticleAnsatz, p::Real, lmps::MPSMultiline,
                     lenvs=environments(lmps, H), rmps=lmps,
                     renvs=lmps === rmps ? lenvs : environments(rmps, H);
                     sector=one(sectortype(lmps)), num=1, solver=Defaults.linearsolver)
    V₀ = Multiline(map(1:size(lmps, 1)) do row
                       return LeftGaugedQP(rand, lmps[row], rmps[row]; sector, momentum=p)
                   end)

    return excitations(H, alg, V₀, lenvs, renvs; num=num, solver=solver)
end

################################################################################
#                                H_eff                                         #
################################################################################

function effective_excitation_hamiltonian(H::MPOHamiltonian, ϕ::QP,
                                          envs=environments(ϕ, H),
                                          energy=effective_excitation_renormalization_energy(H,
                                                                                             ϕ,
                                                                                             envs.lenvs,
                                                                                             envs.renvs))
    ϕ′ = similar(ϕ)

    @static if Defaults.parallelize_sites
        @sync for loc in 1:length(ϕ)
            Threads.@spawn begin
                ϕ′[loc] = _effective_excitation_local_apply(loc, ϕ, H, energy[loc],
                                                            envs)
            end
        end
    else
        for loc in 1:length(ϕ)
            ϕ′[loc] = _effective_excitation_local_apply(loc, ϕ, H, energy[loc], envs)
        end
    end

    return ϕ′
end

function effective_excitation_hamiltonian(H::MPOMultiline, exci::Multiline{<:InfiniteQP},
                                          envs=environments(exci, H))
    toreturn = Multiline(similar.(exci.data))

    left_gs = exci.left_gs
    right_gs = exci.right_gs

    for row in 1:size(H, 1)
        Bs = [exci[row][i] for i in 1:size(H, 2)]
        for col in 1:size(H, 2)
            en = @plansor conj(left_gs.AC[row, col][2 6; 4]) *
                          leftenv(envs.lenvs, row, col, left_gs)[2 5; 3] *
                          left_gs.AC[row + 1, col][3 7; 1] *
                          H[row, col][5 6; 7 8] *
                          rightenv(envs.lenvs, row, col, left_gs)[1 8; 4]

            @plansor T[-1 -2; -3 -4] := leftenv(envs.lenvs, row, col, left_gs)[-1 5; 4] *
                                        Bs[col][4 2; -3 1] *
                                        H[row, col][5 -2; 2 3] *
                                        rightenv(envs.renvs, row, col, right_gs)[1 3; -4]

            @plansor T[-1 -2; -3 -4] += envs.lBs[row, col - 1][-1 4; -3 5] *
                                        right_gs.AR[row, col][5 2; 1] *
                                        H[row, col][4 -2; 2 3] *
                                        rightenv(envs.renvs, row, col, right_gs)[1 3; -4]

            @plansor T[-1 -2; -3 -4] += leftenv(envs.lenvs, row, col, left_gs)[-1 2; 1] *
                                        left_gs.AL[row, col][1 3; 4] *
                                        H[row, col][2 -2; 3 5] *
                                        envs.rBs[row, col + 1][4 5; -3 -4]

            toreturn[row + 1][col] = T / en
        end
    end

    return toreturn
end

function _effective_excitation_local_apply(loc, ϕ, H::MPOHamiltonian, E::Number, envs)
    B = ϕ[loc]
    GL = leftenv(envs.lenvs, loc, ϕ.left_gs)
    GR = rightenv(envs.renvs, loc, ϕ.right_gs)

    # renormalize first -> allocates destination
    B′ = scale(B, -E)

    # add all contributions
    for (j, k) in keys(H[loc])
        h = H[loc][j, k]
        # B in center
        @plansor begin
            B′[-1 -2; -3 -4] += GL[j][-1 5; 4] * B[4 2; -3 1] * h[5 -2; 2 3] *
                                GR[k][1 3; -4]
        end

        # B to the left
        if loc > 1 || ϕ isa InfiniteQP
            AR = ϕ.right_gs.AR[loc]
            GBL = envs.lBs[j, loc]
            @plansor begin
                B′[-1 -2; -3 -4] += GBL[-1 4; -3 5] * AR[5 2; 1] * h[4 -2; 2 3] *
                                    GR[k][1 3; -4]
            end
        end

        # B to the right
        if loc < length(ϕ.left_gs) || ϕ isa InfiniteQP
            AL = ϕ.left_gs.AL[loc]
            GBR = envs.rBs[k, loc]
            @plansor begin
                B′[-1 -2; -3 -4] += GL[j][-1 2; 1] * AL[1 3; 4] * h[2 -2; 3 5] *
                                    GBR[4 5; -3 -4]
            end
        end
    end

    return B′
end

function effective_excitation_renormalization_energy(H, ϕ, lenvs, renvs)
    E_left = map(1:length(ϕ)) do loc
        AC = ϕ.left_gs.AC[loc]
        GL = leftenv(lenvs, loc, ϕ.left_gs)
        GR = rightenv(lenvs, loc, ϕ.left_gs)
        return sum(keys(H[loc]); init=zero(scalartype(ϕ))) do (j, k)
            return @plansor conj(AC[2 6; 4]) * GL[j][2 5; 3] * AC[3 7; 1] *
                            H[loc][j, k][5 6; 7 8] *
                            GR[k][1 8; 4]
        end
    end

    ϕ.trivial && return E_left

    E_right = map(1:length(ϕ)) do loc
        AC = ϕ.right_gs.AC[loc]
        GL = leftenv(renvs, loc, ϕ.right_gs)
        GR = rightenv(renvs, loc, ϕ.right_gs)
        return sum(keys(H[loc]); init=zero(scalartype(ϕ))) do (j, k)
            return @plansor conj(AC[2 6; 4]) * GL[j][2 5; 3] * AC[3 7; 1] *
                            H[loc][j, k][5 6; 7 8] *
                            GR[k][1 8; 4]
        end
    end

    return (E_left .+ E_right) ./ 2
end
