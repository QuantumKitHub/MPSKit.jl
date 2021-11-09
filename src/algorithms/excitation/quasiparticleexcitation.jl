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
    excitations(H, alg::QuasiparticleAnsatz, args...; kwargs...)

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
    H, alg::QuasiparticleAnsatz, V₀::InfiniteQP, lenvs, renvs;
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
    H, alg::QuasiparticleAnsatz, V₀::InfiniteQP, lenvs;
    num=1, solver=Defaults.solver
)
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = V₀.trivial ? lenvs : environments(V₀.right_gs, H, solver=solver)
    return excitations(H, alg, V₀, lenvs, renvs; num, solver)
end
function excitations(
    H, alg::QuasiparticleAnsatz, V₀::InfiniteQP;
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
    H, alg::QuasiparticleAnsatz, p::Real,
    lmps::InfiniteMPS, lenvs=environments(lmps, H),
    rmps::InfiniteMPS=lmps, renvs=lmps===rmps ? lenvs : environments(rmps, H);
    sector=first(sectors(oneunit(virtualspace(lmps, 1)))), num=1,
    solver=Defaults.solver
)
    V₀ = LeftGaugedQP(rand, lmps, rmps; sector=sector, momentum=p)
    return excitations(H, alg, V₀, lenvs, renvs; num=num, solver=solver)
end
function excitations(
    H, alg::QuasiparticleAnsatz, momenta, lmps, lenvs=environments(lmps, H),
    rmps=lmps, renvs=lmps===rmps ? lenvs : environments(rmps, H);
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
    H, alg::QuasiparticleAnsatz,
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
    H, alg::QuasiparticleAnsatz,
    lmps::FiniteMPS, lenvs=environments(lmps, H),
    rmps::FiniteMPS=lmps, renvs=lmps===rmps ? lenvs : environments(rmps, H)environmentsleft;
    sector=first(sectors(oneunit(virtualspace(lmps, 1)))), num=1
)
    V₀ = LeftGaugedQP(rand, lmps, rmps; sector=sector);
    return excitations(H, alg, V₀, lenvs, renvs; num=num)
end

################################################################################
#                           Statmech Excitations                               #
################################################################################

function excitations(
    H::MPOMultiline,alg::QuasiparticleAnsatz,V₀::Multiline{<:InfiniteQP},lenvs,renvs;
    num = 1, solver = Defaults.solver)

    qp_envs(V) = environments(V, H, lenvs, renvs, solver=solver)
    function H_eff(oV)
        V = Multiline(oV.vecs);
        RecursiveVec(effective_excitation_hamiltonian(H, V, qp_envs(V)).data.data)
    end

    Es, Vs, convhist = eigsolve(
        H_eff, RecursiveVec(V₀.data.data), num, :LM, tol=alg.toler, krylovdim=alg.krylovdim
    )

    convhist.converged < num &&
        @warn "Quasiparticle didn't converge: $(convhist.normres)"

    return Es, map(x->Multiline(x.vecs),Vs)
end

function excitations(
    H::MPOMultiline, alg::QuasiparticleAnsatz, V₀::Multiline{<:InfiniteQP}, lenvs;
    num=1, solver=Defaults.solver
)
    # Infer `renvs` in function body as it depends on `solver`.
    renvs = V₀.trivial ? lenvs : environments(V₀.right_gs, H, solver=solver)
    return excitations(H, alg, V₀, lenvs, renvs; num, solver)
end
function excitations(
    H::MPOMultiline, alg::QuasiparticleAnsatz, V₀::Multiline{<:InfiniteQP};
    num=1, solver=Defaults.solver
)
    # Infer `lenvs` in function body as it depends on `solver`.
    lenvs = environments(V₀.left_gs, H, solver=solver)
    return excitations(H, alg, V₀, lenvs; num, solver)
end


function excitations(
    H::DenseMPO, alg::QuasiparticleAnsatz, p::Real,
    lmps, lenvs=environments(lmps, H),
    rmps=lmps, renvs=lmps===rmps ? lenvs : environments(rmps, H);
    sector=first(sectors(oneunit(virtualspace(lmps, 1)))), num=1,
    solver=Defaults.solver
)
    multiline_lmps = convert(MPSMultiline,lmps);
    if lmps === rmps
        excitations(convert(MPOMultiline,H),alg,p,multiline_lmps,lenvs,multiline_lmps,
            lenvs;sector,num,solver);
    else
        excitations(convert(MPOMultiline,H),alg,p,multiline_lmps,lenvs,convert(MPSMultiline,rmps),
            renvs;sector,num,solver);
    end
end

function excitations(
    H::MPOMultiline, alg::QuasiparticleAnsatz, p::Real,
    lmps::MPSMultiline, lenvs=environments(lmps, H),
    rmps=lmps, renvs=lmps===rmps ? lenvs : environments(rmps, H);
    sector=first(sectors(oneunit(virtualspace(lmps, 1)))), num=1,
    solver=Defaults.solver
)
    V₀ = Multiline(map(1:size(lmps,1)) do row
        LeftGaugedQP(rand,lmps[row],rmps[row];sector,momentum=p)
    end)

    return excitations(H, alg, V₀, lenvs, renvs; num=num, solver=solver)
end

################################################################################
#                                H_eff                                         #
################################################################################

function effective_excitation_hamiltonian(
    H::MPOHamiltonian, exci::QP,envs=environments(exci, H)
)
    Bs = [exci[i] for i in 1:length(exci)];
    toret = similar(exci);

    #do necessary contractions
    for i = 1:length(exci)
        T = zero(Bs[i]);

        for (j,k) in keys(H[i])
            @plansor T[-1 -2;-3 -4] +=    leftenv(envs.lenvs,i,exci.left_gs)[j][-1 5;4]*
                                                Bs[i][4 2;-3 1]*
                                                H[i][j,k][5 -2;2 3]*
                                                rightenv(envs.renvs,i,exci.right_gs)[k][1 3;-4]

            # <B|H|B>-<H>
            en = @plansor    conj(exci.left_gs.AC[i][2 6;4])*
                            leftenv(envs.lenvs,i,exci.left_gs)[j][2 5;3]*
                            exci.left_gs.AC[i][3 7;1]*
                            H[i][j,k][5 6;7 8]*
                            rightenv(envs.lenvs,i,exci.left_gs)[k][1 8;4]

            T -= Bs[i]*en
            if i>1 || exci isa InfiniteQP
                @plansor T[-1 -2;-3 -4] +=    envs.lBs[i-1][j][-1 4;-3 5]*
                                                    exci.right_gs.AR[i][5 2;1]*
                                                    H[i][j,k][4 -2;2 3]*
                                                    rightenv(envs.renvs,i,exci.right_gs)[k][1 3;-4]
            end
            if i<length(exci.left_gs) || exci isa InfiniteQP
                @plansor T[-1 -2;-3 -4] +=    leftenv(envs.lenvs,i,exci.left_gs)[j][-1 2;1]*
                                                    exci.left_gs.AL[i][1 3;4]*
                                                    H[i][j,k][2 -2;3 5]*
                                                    envs.rBs[i+1][k][4 5;-3 -4]
            end
        end

        toret[i] = T;

    end

    return toret

end

function effective_excitation_hamiltonian(
    H::MPOMultiline, exci::Multiline{<:InfiniteQP},envs=environments(exci, H)
)

    toreturn = Multiline(similar.(exci.data));

    left_gs = exci.left_gs;
    right_gs = exci.right_gs;

    for row in 1:size(H,1)
        Bs = [exci[row][i] for i in 1:size(H,2)];
        for col in 1:size(H,2)
            en = @plansor    conj(left_gs.AC[row,col][2 6;4])*
                            leftenv(envs.lenvs,row,col,left_gs)[2 5;3]*
                            left_gs.AC[row+1,col][3 7;1]*
                            H[row,col][5 6;7 8]*
                            rightenv(envs.lenvs,row,col,left_gs)[1 8;4]

            @plansor T[-1 -2;-3 -4] := leftenv(envs.lenvs,row,col,left_gs)[-1 5;4]*
                                                Bs[col][4 2;-3 1]*
                                                H[row,col][5 -2;2 3]*
                                                rightenv(envs.renvs,row,col,right_gs)[1 3;-4]


            @plansor T[-1 -2;-3 -4] +=    envs.lBs[row][col-1][-1 4;-3 5]*
                                                right_gs.AR[row,col][5 2;1]*
                                                H[row,col][4 -2;2 3]*
                                                rightenv(envs.renvs,row,col,right_gs)[1 3;-4]

            @plansor T[-1 -2;-3 -4] +=    leftenv(envs.lenvs,row,col,left_gs)[-1 2;1]*
                                                left_gs.AL[row,col][1 3;4]*
                                                H[row,col][2 -2;3 5]*
                                                envs.rBs[row][col+1][4 5;-3 -4]

            toreturn[row+1][col] = T/en;
        end
    end

    return toreturn;
end
