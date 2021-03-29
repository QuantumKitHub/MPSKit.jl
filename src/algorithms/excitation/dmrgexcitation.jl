@with_kw struct FiniteExcited{A} <: Algorithm
    gsalg::A = Dmrg()
    weight::Float64 = 10.0;
end


function excitations(hamiltonian::MPOHamiltonian,alg::FiniteExcited,states::Vector{T};init = FiniteMPS([copy(first(states).AC[i]) for i in 1:length(first(states))]),num = 1) where T <:FiniteMPS
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

    overlaps  = map(projectout) do st
        @tensor leftstart[-1 -2 -3;-4] := l_LL(st)[-3,-4]*l_LL(state)[-1,-2]
        @tensor rightstart[-1 -2 -3;-4] := r_RR(st)[-1,-2]*r_RR(state)[-3,-4]
        environments(state,st,leftstart,rightstart)
    end

    FinExEnv(weight,overlaps,hamenv)
end


function ac_prime(x::MPSTensor,pos::Int,mps::Union{FiniteMPS,MPSComoving},cache::FinExEnv)
    y = ac_prime(x,pos,mps,cache.hamenv)

    for i in cache.overlaps
        @tensor v[-1 -2;-3 -4] := leftenv(i,pos,mps)[4,-1,-2,5]*i.above.AC[pos][5,2,1]*rightenv(i,pos,mps)[1,-3,-4,3]*conj(x[4,2,3])
        @tensor y[-1 -2;-3] += conj(v[1,2,5,6])*(cache.weight*leftenv(i,pos,mps))[-1,1,2,4]*i.above.AC[pos][4,-2,3]*rightenv(i,pos,mps)[3,5,6,-3]
    end

    y
end
function ac2_prime(x::MPOTensor,pos::Int,mps::Union{FiniteMPS,MPSComoving},cache::FinExEnv)
    y = ac2_prime(x,pos,mps,cache.hamenv)

    for i in cache.overlaps
        @tensor v[-1 -2;-3 -4] := leftenv(i,pos,mps)[6,-1,-2,7]*i.above.AC[pos][7,4,5]*i.above.AR[pos+1][5,2,1]*rightenv(i,pos+1,mps)[1,-3,-4,3]*conj(x[6,4,2,3])
        @tensor y[-1 -2;-3 -4] += conj(v[2,3,5,6])*(cache.weight*leftenv(i,pos,mps))[-1,2,3,4]*i.above.AC[pos][4,-2,7]*i.above.AR[pos+1][7,-3,1]*rightenv(i,pos+1,mps)[1,5,6,-4]
    end

    y
end
