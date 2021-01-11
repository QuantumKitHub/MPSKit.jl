#=
    an excitation tensor has 4 legs (1,2),(3,4)
    the first and the last are virtual, the second is physical, the third is the utility leg
=#
@with_kw struct QuasiparticleAnsatz <: Algorithm
    toler::Float64 = 1e-10;
    krylovdim::Int = 30;
end


include("excitransfers.jl")

"
    quasiparticle_excitation calculates the energy of the first excited state at momentum 'moment'
"
function excitations(hamiltonian::Hamiltonian, alg::QuasiparticleAnsatz, momentum::Float64, mpsleft::InfiniteMPS, environmentsleft=environments(mpsleft,hamiltonian), mpsright::InfiniteMPS=mpsleft, environmentsright=environmentsleft;
    sector = first(sectors(oneunit(virtualspace(mpsleft,1)))), num=1)

    V_initial = rand_quasiparticle(mpsleft,mpsright;sector=sector,momentum=momentum);

    #the function that maps x->B and then places this in the excitation hamiltonian
    eigEx(V) = effective_excitation_hamiltonian(hamiltonian, V, environments(V,hamiltonian,environmentsleft, environmentsright))
    Es,Vs,convhist = eigsolve(eigEx, V_initial, num, :SR, tol=alg.toler,krylovdim=alg.krylovdim)
    convhist.converged<num && @warn "quasiparticle didn't converge k=$(momentum) $(convhist.normres)"

    return Es,Vs
end

#pretty much identical to the infinite mps code, except for the lack of momentum label
function excitations(hamiltonian::Hamiltonian, alg::QuasiparticleAnsatz, mpsleft::FiniteMPS, environmentsleft=environments(mpsleft,hamiltonian), mpsright::FiniteMPS=mpsleft, environmentsright=environmentsleft;
    sector = first(sectors(oneunit(virtualspace(mpsleft,1)))),num=1)

    V_initial = rand_quasiparticle(mpsleft,mpsright;sector=sector);

    #the function that maps x->B and then places this in the excitation hamiltonian
    eigEx(V) = effective_excitation_hamiltonian(hamiltonian, V, environments(V,hamiltonian,environmentsleft, environmentsright))
    Es,Vs,convhist = eigsolve(eigEx, V_initial, num, :SR, tol=alg.toler,krylovdim=alg.krylovdim)
    convhist.converged<num && @warn "quasiparticle didn't converge $(convhist.normres)"

    return Es,Vs
end

#give it a vector of momentum points
function excitations(hamiltonian::Hamiltonian, alg::QuasiparticleAnsatz, momenta::AbstractVector, mpsleft::InfiniteMPS, environmentsleft=environments(mpsleft,hamiltonian), mpsright::InfiniteMPS=mpsleft, environmentsright=environmentsleft;
    num=1,verbose=Defaults.verbose,kwargs...)

    tasks = map(enumerate(momenta)) do (i,p)
        @Threads.spawn begin
            (E,V) = excitations(hamiltonian,alg, p, mpsleft, environmentsleft, mpsright, environmentsright; num=num,kwargs...)
            verbose && @info "Found excitations for p = $(p)"
            (E,V)
        end
    end

    fetched = fetch.(tasks);

    Ep = permutedims(reduce(hcat,map(x->x[1][1:num],fetched)));
    Bp = permutedims(reduce(hcat,map(x->x[2][1:num],fetched)));

    return Ep,Bp
end

function effective_excitation_hamiltonian(ham::MPOHamiltonian, exci::QP,envs=environments(exci,ham))
    odim = ham.odim;

    Bs = [exci[i] for i in 1:length(exci)];
    toret = zero.(Bs);

    #do necessary contractions
    for i = 1:length(exci)
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    leftenv(envs.lenvs,i,exci.left_gs)[j][-1,1,2]*
                                                Bs[i][2,3,-3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rightenv(envs.renvs,i,exci.right_gs)[k][4,5,-4]

            # <B|H|B>-<H>
            en = @tensor    conj(exci.left_gs.AC[i][11,12,13])*
                            leftenv(envs.lenvs,i,exci.left_gs)[j][11,1,2]*
                            exci.left_gs.AC[i][2,3,4]*
                            ham[i,j,k][1,12,5,3]*
                            rightenv(envs.lenvs,i,exci.left_gs)[k][4,5,13]

            toret[i] -= Bs[i]*en
            if i>1 || exci isa InfiniteQP
                @tensor toret[i][-1,-2,-3,-4] +=    envs.lBs[mod1(i-1,end)][j][-1,1,-3,2]*
                                                    exci.right_gs.AR[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    rightenv(envs.renvs,i,exci.right_gs)[k][4,5,-4]
            end
            if i<length(exci.left_gs) || exci isa InfiniteQP
                @tensor toret[i][-1,-2,-3,-4] +=    leftenv(envs.lenvs,i,exci.left_gs)[j][-1,1,2]*
                                                    exci.left_gs.AL[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    envs.rBs[mod1(i+1,end)][k][4,-3,5,-4]
            end
        end

    end

    toret_vec = similar(exci);
    for i in 1:length(exci)
        toret_vec[i] = toret[i]
    end
    return toret_vec

end
