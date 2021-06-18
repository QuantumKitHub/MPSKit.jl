#=
    an excitation tensor has 4 legs (1,2),(3,4)
    the first and the last are virtual, the second is physical, the third is the utility leg
=#
@with_kw struct QuasiparticleAnsatz <: Algorithm
    toler::Float64 = 1e-10;
    krylovdim::Int = 30;
end


include("excitransfers.jl")

"""
    excitations(H::Hamiltonian, alg::QuasiparticleAnsatz, args...; kwargs...)

Compute the first excited states and their energy gap above a groundstate.

This is an implementation of the algorithm found [here](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.080401).
"""
function excitations end

################################################################################
#                           Infinite Excitations                               #
################################################################################
"""
    excitations(H, alg, V₀::InfiniteQP, lenvs, renvs; num, solver)

Optimise the infinite quasiparticle state ```V₀```.

# Arguments
- `V₀::InfiniteQP`: initial quasiparticle state.
- `lenvs=environments(V₀.left_gs, H; solver=solver)`: left environment of the groundstate.
- `renvs=(V₀.trivial ? lenvs : environments(V₀.right_gs, H; solver=solver))`: right environment of the groundstate.
- `num=1`: number of excited states to compute.
- `solver=Defaults.solver`: the algorithm to compute the environments.
"""
function excitations(
    H::Hamiltonian, alg::QuasiparticleAnsatz, V₀::InfiniteQP, lenvs, renvs;
    num=1, solver=Defaults.solver
)
    qp_envs(V) = environments(V, H, lenvs, renvs, solver=solver)
    H_eff = @closure(V->effective_excitation_hamiltonian(H, V, qp_envs(V)))
    
    Es, Vs, convhist = eigsolve(
        H_eff, V₀, num, :SR, tol=alg.toler, krylovdim=alg.krylovdim
    )
    
    convhist.converged < num && 
        @warn "Quasiparticle didn't converge: $(convhist.normres)"
    
    return Es, Vs
end
function excitations(
    H::Hamiltonian, alg::QuasiparticleAnsatz, V₀::InfiniteQP, lenvs;
    num=1, solver=Defaults.solver
)
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = V₀.trivial ? lenvs : environments(V₀.right_gs, H, solver=solver)
    return excitations(H, alg, V₀, lenvs, renvs; num, solver)
end
function excitations(
    H::Hamiltonian, alg::QuasiparticleAnsatz, V₀::InfiniteQP;
    num=1, solver=Defaults.solver
)
    # Infer `lenvs` in function body as it depends on `solver`.
    lenvs = environments(V₀.left_gs, H, solver=solver)
    return excitations(H, alg, V₀, lenvs; num, solver)
end

"""
    excitations(H, alg, p::Real, lmps, lenvs, rmps, renvs; sector, kwargs...)

Create and optimise an infinite quasiparticle state with momentum ```p```.
"""
function excitations(
    H::Hamiltonian, alg::QuasiparticleAnsatz, p::Real, 
    lmps::InfiniteMPS, lenvs=environments(lmps, H),
    rmps::InfiniteMPS=lmps, renvs=lmps===rmps ? lenvs : environments(rmps, H);
    sector=first(sectors(oneunit(virtualspace(lmps, 1)))), num=1,
    solver=Defaults.solver
)
    V₀ = LeftGaugedQP(rand, lmps, rmps; sector=sector, momentum=p)
    return excitations(H, alg, V₀, lenvs, renvs; num=num, solver=solver)
end
function excitations(
    H::Hamiltonian, alg::QuasiparticleAnsatz, momenta::AbstractVector,
    lmps::InfiniteMPS, lenvs=environments(lmps, H),
    rmps::InfiniteMPS=lmps, renvs=lmps===rmps ? lenvs : environments(rmps, H);
    verbose=Defaults.verbose, num=1, solver=Defaults.solver, sector=first(sectors(oneunit(virtualspace(lmps, 1))))
)
    tasks = map(momenta) do p
        @Threads.spawn begin
            (E, V) = excitations(H, alg, p, lmps, lenvs, rmps, renvs; num=num, solver=solver, sector=sector)
            verbose && @info "Found excitations for p = $(p)"
            (E, V)
        end
    end
    
    fetched = fetch.(tasks);
    
    Ep = permutedims(reduce(hcat, map(x->x[1][1:num], fetched)));
    Bp = permutedims(reduce(hcat, map(x->x[2][1:num], fetched)));
    
    return Ep, Bp
end

################################################################################
#                           Finite Excitations                                 #
################################################################################
"""
    excitations(H, alg, V₀::FiniteQP[, lenvs[, renvs]]; num)

Optimise the finite quasiparticle state ```V₀```.

# Arguments
- `V₀::FiniteQP`: initial quasiparticle state.
- `lenvs=environments(V₀.left_gs, H)`: left environment of the groundstate.
- `renvs=(V₀.trivial ? lenvs : environments(V₀.right_gs, H))`: right environment of the groundstate.
- `num=1`: number of excited states to compute.
"""
function excitations(
    H::Hamiltonian, alg::QuasiparticleAnsatz, 
    V₀::FiniteQP, lenvs=environments(V₀.left_gs, H),
    renvs=V₀.trivial ? lenvs : environments(V₀.right_gs, H);
    num=1
)
    qp_envs(V) = environments(V, H, lenvs, renvs)
    H_eff = @closure(V->effective_excitation_hamiltonian(H, V, qp_envs(V)))
    
    Es, Vs, convhist = eigsolve(
        H_eff, V₀, num, :SR, tol=alg.toler, krylovdim=alg.krylovdim
    )
    
    convhist.converged < num &&
        @warn "Quasiparticle didn't converge: $(convhist.normres)"
    
    return Es, Vs
end

"""
    excitations(H, alg, [lmps, [lenvs, [rmps, [renvs]]]]; sector, kwargs...)

Create and optimise a finite quasiparticle state.
"""
function excitations(
    H::Hamiltonian, alg::QuasiparticleAnsatz, 
    lmps::FiniteMPS, lenvs=environments(lmps, H),
    rmps::FiniteMPS=lmps, renvs=lmps===rmps ? lenvs : environments(rmps, H)environmentsleft;
    sector=first(sectors(oneunit(virtualspace(lmps, 1)))), num=1
)
    V₀ = LeftGaugedQP(rand, lmps, rmps; sector=sector);
    return excitations(H, alg, V₀, lenvs, renvs; num=num)
end

function effective_excitation_hamiltonian(
    H::MPOHamiltonian, exci::QP,envs=environments(exci, H)
)
    Bs = [exci[i] for i in 1:length(exci)];
    toret = similar(exci);

    #do necessary contractions
    for i = 1:length(exci)
        T = zero(Bs[i]);

        for (j,k) in keys(H,i)
            @tensor T[-1,-2,-3,-4] +=    leftenv(envs.lenvs,i,exci.left_gs)[j][-1,1,2]*
                                                Bs[i][2,3,-3,4]*
                                                H[i,j,k][1,-2,5,3]*
                                                rightenv(envs.renvs,i,exci.right_gs)[k][4,5,-4]

            # <B|H|B>-<H>
            en = @tensor    conj(exci.left_gs.AC[i][11,12,13])*
                            leftenv(envs.lenvs,i,exci.left_gs)[j][11,1,2]*
                            exci.left_gs.AC[i][2,3,4]*
                            H[i,j,k][1,12,5,3]*
                            rightenv(envs.lenvs,i,exci.left_gs)[k][4,5,13]

            T -= Bs[i]*en
            if i>1 || exci isa InfiniteQP
                @tensor T[-1,-2,-3,-4] +=    envs.lBs[i-1][j][-1,1,-3,2]*
                                                    exci.right_gs.AR[i][2,3,4]*
                                                    H[i,j,k][1,-2,5,3]*
                                                    rightenv(envs.renvs,i,exci.right_gs)[k][4,5,-4]
            end
            if i<length(exci.left_gs) || exci isa InfiniteQP
                @tensor T[-1,-2,-3,-4] +=    leftenv(envs.lenvs,i,exci.left_gs)[j][-1,1,2]*
                                                    exci.left_gs.AL[i][2,3,4]*
                                                    H[i,j,k][1,-2,5,3]*
                                                    envs.rBs[i+1][k][4,-3,5,-4]
            end
        end

        toret[i] = T;

    end

    return toret

end
