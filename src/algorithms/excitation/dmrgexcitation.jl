@with_kw struct FiniteExcited{A} <: Algorithm
    gsalg::A = Dmrg()
    weight::Float64 = 10.0;
end


function excitations(hamiltonian::MPOHamiltonian,alg::FiniteExcited,states::Vector{T};init = FiniteMPS([copy(first(states).AC[i]) for i in 1:length(first(gs))]),num = 1) where T <:FiniteMPS
    if num == 0
        return (eltype(eltype(T))[],T[])
    end

    envs = environments(init,hamiltonian,alg.weight,states);
    (ne,_) = find_groundstate(init,hamiltonian,alg.gsalg,envs);

    push!(states,ne);

    (ens,excis) = excitations(hamiltonian,alg,states;init = init,num = num-1);

    push!(ens,sum(expectation_value(ne,hamiltonian)))
    push!(excis,ne);

    return ens,excis
end
excitations(hamiltonian::Hamiltonian, alg::FiniteExcited,gs::FiniteMPS;kwargs...) = excitations(hamiltonian,alg,[gs];kwargs...)

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
