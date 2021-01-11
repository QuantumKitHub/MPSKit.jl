@with_kw struct FiniteExcited{A} <: Algorithm
    gsalg::A = Dmrg()
    weight::Float64 = 10.0;
end



function excitations(hamiltonian::Hamiltonian, alg::FiniteExcited,gs::FiniteMPS,envs = environments(gs,hamiltonian);init = FiniteMPS([copy(gs.AC[i]) for i in 1:length(gs)]),num = 1)
    #iteratively call alg.gsalg
    found = [gs];
    energies = [sum(expectation_value(gs,hamiltonian))];

    for i in 1:num
        envs = environments(init,hamiltonian,alg.weight,found);

        (ne,_) = find_groundstate(init,hamiltonian,alg.gsalg,envs);

        push!(found,ne);
        push!(energies,sum(expectation_value(ne,hamiltonian)));
    end

    return energies[2:end].-energies[1],found[2:end];
end

#some simple environments

#=
FinExEnv == FinEnv(state,ham) + weight * sum_i <state|psi_i> <psi_i|state>
=#
struct FinExEnv{A,B} <:Cache
    weight::Float64
    overlaps::Vector{A}
    hamenv::B
end

function environments(state,ham,weight,projectout::Vector)
    hamenv = environments(state,ham);
    overlaps = [environments(pr,state) for pr in projectout];

    FinExEnv(weight,overlaps,hamenv)
end


function ac_prime(x::MPSTensor,pos::Int,mps::Union{FiniteMPS,MPSComoving},cache::FinExEnv)
    y = ac_prime(x,pos,mps,cache.hamenv)

    for i in cache.overlaps
        v = @tensor leftenv(i,pos,mps)[1,2]*i.above.AC[pos][2,3,4]*rightenv(i,pos,mps)[4,5]*conj(x[1,3,5])
        @tensor y[-1 -2;-3]+= (v'*cache.weight * leftenv(i,pos,mps))[-1,1]*i.above.AC[pos][1,-2,2]*rightenv(i,pos,mps)[2,-3]
    end

    y
end
function ac2_prime(x::MPSTensor,pos::Int,mps::Union{FiniteMPS,MPSComoving},cache::FinExEnv)
    y = ac2_prime(x,pos,mps,cache.hamenv)

    for i in cache.overlaps
        v = @tensor leftenv(i,pos,mps)[1,2]*i.above.AC[pos][2,3,4]*i.above.AR[pos+1][4,5,6]*rightenv(i,pos+1,mps)[6,7]*conj(x[1,3,5,7])
        @tensor y[-1 -2;-3 -4]+= (v'*cache.weight * leftenv(i,pos,mps))[-1,1]*i.above.AC[pos][1,-2,2]*i.above.AR[pos+1][2,-3,3]*rightenv(i,pos+1,mps)[3,-4]
    end

    y
end
