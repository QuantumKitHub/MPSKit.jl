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
function quasiparticle_excitation(hamiltonian::Hamiltonian, moment::Float64, mpsleft::InfiniteMPS, paramsleft=params(mpsleft,hamiltonian), mpsright::InfiniteMPS=mpsleft, paramsright=paramsleft;
    excitation_space=oneunit(space(mpsleft.AL[1],1)), num=1 , toler = 1e-10,krylovdim=30)

    V_initial = rand_quasiparticle(moment,mpsleft,mpsright;excitation_space);

    #the function that maps x->B and then places this in the excitation hamiltonian
    eigEx(V) = effective_excitation_hamiltonian(hamiltonian, V, paramsleft, paramsright)
    Es,Vs,convhist = eigsolve(eigEx, V_initial, num, :SR, tol=toler,krylovdim=krylovdim)
    convhist.converged<num && @warn "quasiparticle didn't converge k=$(moment) $(convhist.normres)"

    return Es,Vs
end

#pretty much identical to the infinite mps code, except for the lack of momentum label
function quasiparticle_excitation(hamiltonian::Hamiltonian, mpsleft::FiniteMPS, paramsleft=params(mpsleft,hamiltonian), mpsright::FiniteMPS=mpsleft, paramsright=paramsleft;
    excitation_space=oneunit(space(mpsleft.AL[1],1)),num=1, toler = 1e-10,krylovdim=30)

    V_initial = rand_quasiparticle(mpsleft,mpsright;excitation_space);

    #the function that maps x->B and then places this in the excitation hamiltonian
    eigEx(V) = effective_excitation_hamiltonian(hamiltonian, V, paramsleft, paramsright)
    Es,Vs,convhist = eigsolve(eigEx, V_initial, num, :SR, tol=toler,krylovdim=krylovdim)
    convhist.converged<num && @warn "quasiparticle didn't converge $(convhist.normres)"

    return Es,Vs
end


function quasiparticle_excitation(hamiltonian::Hamiltonian, momenta::AbstractVector, mpsleft::InfiniteMPS, paramsleft=params(mpsleft,hamiltonian), mpsright::InfiniteMPS=mpsleft, paramsright=paramsleft;
    num=1,verbose=Defaults.verbose,kwargs...)

    tasks = map(enumerate(momenta)) do (i,p)
        @Threads.spawn begin
            (E,V) = quasiparticle_excitation(hamiltonian, p, mpsleft, paramsleft, mpsright, paramsright; num=num,kwargs...)
            verbose && println("Found excitations for p = $(p)")
            (E,V)
        end
    end

    fetched = fetch.(tasks);

    Ep = permutedims(reduce(hcat,map(x->x[1][1:num],fetched)));
    Bp = permutedims(reduce(hcat,map(x->x[2][1:num],fetched)));

    return Ep,Bp
end
