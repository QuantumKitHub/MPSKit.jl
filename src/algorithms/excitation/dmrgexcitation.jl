@with_kw struct FiniteExcited{A} <: Algorithm
    gsalg::A = Dmrg()
    weight::Float64 = 10.0;
end


function excitations(hamiltonian::MPOHamiltonian,alg::FiniteExcited,states::Vector{T};init = FiniteMPS([copy(first(states).AC[i]) for i in 1:length(first(states))]),num = 1) where T <:FiniteMPS
    num == 0 && return (eltype(eltype(T))[],T[])

    envs = environments(init,hamiltonian,alg.weight,states);
    (ne,_) = find_groundstate(init,hamiltonian,alg.gsalg,envs);

    push!(states,ne);

    (ens,excis) = excitations(hamiltonian,alg,states;init = init,num = num-1);

    push!(ens,sum(expectation_value(ne,hamiltonian)))
    push!(excis,ne);

    return ens,excis
end
excitations(hamiltonian, alg::FiniteExcited,gs::FiniteMPS;kwargs...) = excitations(hamiltonian,alg,[gs];kwargs...)

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
        @plansor leftstart[-1;-2 -3 -4] := l_LL(st)[-3;-4]*l_LL(state)[-1;-2]
        @plansor rightstart[-1;-2 -3 -4] := r_RR(st)[-1;-2]*r_RR(state)[-3;-4]
        environments(state,fill(nothing,length(st)),st,leftstart,rightstart)
    end

    FinExEnv(weight,overlaps,hamenv)
end

leftenv(ca::FinExEnv,pos::Int,st::Union{FiniteMPS,MPSComoving}) =
    (leftenv(ca.hamenv,pos,st),[leftenv(i,pos,st) for i in 1:length(ca.overlaps)]);

rightenv(ca::FinExEnv,pos::Int,st::Union{FiniteMPS,MPSComoving}) =
    (rightenv(ca.hamenv,pos,st),[rightenv(i,pos,st) for i in 1:length(ca.overlaps)]);


function ac_prime(x::MPSTensor,opp,leftenv,rightenv)
    lh = first(leftenv);
    rh = first(rightenv);

    y = AC_eff(pos,opp,lh,rh)*x

    for (ind,i) in enumerate(cache.overlaps)
        le = leftenv[2][ind];
        re = rightenv[2][ind];

        @plansor v[-1;-2 -3 -4] := le[4;-1 -2 5]*i.above.AC[pos][5 2;1]*re[1;-3 -4 3]*conj(x[4 2;3])
        @plansor y[-1 -2;-3] += conj(v[1;2 5 6])*(cache.weight*le)[-1;1 2 4]*i.above.AC[pos][4 -2;3]*re[3;5 6 -3]
    end

    y
end
function ac2_prime(x::MPOTensor,opp,leftenv,rightenv)
    lh = first(leftenv);
    rh = first(rightenv);

    y = AC2_eff(pos,opp,lh,rh)*x

    for (ind,i) in enumerate(cache.overlaps)
        le = leftenv[2][ind];
        re = rightenv[2][ind];

        @plansor v[-1;-2 -3 -4] := le[6;-1 -2 7]*i.above.AC[pos][7 4;5]*i.above.AR[pos+1][5 2;1]*re[1;-3 -4 3]*conj(x[6 4;3 2])
        @plansor y[-1 -2;-3 -4] += conj(v[2;3 5 6])*(cache.weight*le)[-1;2 3 4]*i.above.AC[pos][4 -2;7]*i.above.AR[pos+1][7 -4;1]*re[1;5 6 -3]
    end

    y
end
