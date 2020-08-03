#=
    an excitation tensor has 4 legs (1,2),(3,4)
    the first and the last are virtual, the second is physical, the third is the utility leg
=#

include("excitransfers.jl")
include("heff_infinite.jl")
include("heff_finite.jl")
"
    quasiparticle_excitation calculates the energy of the first excited state at momentum 'moment'
"
function quasiparticle_excitation(hamiltonian::Hamiltonian, moment::Float64, mpsleft::InfiniteMPS, paramsleft, mpsright::InfiniteMPS=mpsleft, paramsright=paramsleft;
    excitation_space=oneunit(space(mpsleft.AL[1],1)), trivial=true, num=1 ,X_initial=nothing, toler = 1e-10,krylovdim=30)

    #check whether the provided mps is sensible
    @assert length(mpsleft) == length(mpsright)

    #find the left null spaces for the TNS
    LeftNullSpace = [adjoint(rightnull(adjoint(v))) for v in mpsleft.AL]

    #we need an initial array of Xs, one for every site in unit cell which the user may of may not have provided by the user
    if X_initial == nothing
        X_initial=[TensorMap(rand,eltype(mpsleft),space(LeftNullSpace[loc],3)',excitation_space'*space(mpsright.AR[ loc+1],1)) for loc in 1:length(mpsleft)]
    end

    #the function that maps x->B and then places this in the excitation hamiltonian
    function eigEx(rec)
        x = rec.vecs

        B = [ln*cx for (ln,cx) in zip(LeftNullSpace,x)]

        Bseff = effective_excitation_hamiltonian(trivial, hamiltonian, B, moment, mpsleft, paramsleft, mpsright, paramsright)

        out = [adjoint(ln)*cB for (ln,cB) in zip(LeftNullSpace,Bseff)]

        return RecursiveVec(out)
    end
    Es,Vs,convhist = eigsolve(eigEx, RecursiveVec(X_initial), num, :SR, tol=toler,krylovdim=krylovdim)
    convhist.converged<num && @info "quasiparticle didn't converge k=$(moment) $(convhist.normres)"

    #we dont want to return a RecursiveVec to the user so upack it
    Xs = [v.vecs for v in Vs]
    Bs = [[ln*cx for (ln,cx) in zip(LeftNullSpace,x)] for x in Xs]

    return Es,Bs, Xs
end

function quasiparticle_excitation(hamiltonian::Hamiltonian, momenta::AbstractVector, mpsleft::InfiniteMPS, paramsleft, mpsright::InfiniteMPS=mpsleft, paramsright=paramsleft;
    num=1,verbose=Defaults.verbose,kwargs...)

    tasks = map(enumerate(momenta)) do (i,p)
        @Threads.spawn begin
            verbose && println("Finding excitations for p = $(p)")
            quasiparticle_excitation(hamiltonian, p, mpsleft, paramsleft, mpsright, paramsright; num=num,kwargs...)
        end
    end

    fetched = fetch.(tasks);

    Ep = reduce(hcat,map(x->x[1][1:num],fetched));
    Bp = reduce(hcat,map(x->x[2][1:num],fetched));
    Xp = reduce(hcat,map(x->x[3][1:num],fetched));

    return Ep,Bp,Xp
end

function quasiparticle_excitation(hamiltonian::Hamiltonian, mpsleft::FiniteMPS, paramsleft, mpsright::FiniteMPS=mpsleft, paramsright=paramsleft;
    excitation_space=oneunit(space(mpsleft.AL[1],1)),num=1, toler = 1e-10,krylovdim=30)

    #check whether the provided mps is sensible
    @assert length(mpsleft) == length(mpsright)

    #find the left null spaces for the TNS
    LeftNullSpace = [adjoint(rightnull(adjoint(v))) for v in mpsleft.AL]
    X_initial =[TensorMap(rand,eltype(eltype(mpsleft)),space(LeftNullSpace[loc],3)',excitation_space'*space(mpsright.AR[ loc],3)') for loc in 1:length(mpsleft)]

    #the function that maps x->B and then places this in the excitation hamiltonian
    function eigEx(rec)
        x = rec.vecs

        B = [ln*cx for (ln,cx) in zip(LeftNullSpace,x)]

        Bseff = effective_excitation_hamiltonian(hamiltonian, B, mpsleft, paramsleft, mpsright, paramsright)

        out = [adjoint(ln)*cB for (ln,cB) in zip(LeftNullSpace,Bseff)]

        return RecursiveVec(out)
    end
    Es,Vs,convhist = eigsolve(eigEx, RecursiveVec(X_initial), num, :SR, tol=toler,krylovdim=krylovdim)
    convhist.converged<num && @info "quasiparticle didn't converge k=$(moment) $(convhist.normres)"

    #we dont want to return a RecursiveVec to the user so upack it
    Xs = [v.vecs for v in Vs]
    Bs = [[ln*cx for (ln,cx) in zip(LeftNullSpace,x)] for x in Xs]

    return Es,Bs, Xs
end
