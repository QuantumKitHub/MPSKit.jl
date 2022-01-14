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
struct FinExEnv{A,B,O} <:Cache
    opp::O
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

    FinExEnv(ham,weight,overlaps,hamenv)
end

struct FinEx_AC_eff{S,E}
    pos::Int
    state::S
    envs::E
end


struct FinEx_AC2_eff{S,E}
    pos::Int
    state::S
    envs::E
end

AC_eff(pos::Int,mps::Union{FiniteMPS,MPSComoving},cache::FinExEnv) = FinEx_AC_eff(pos,mps,cache)
AC2_eff(pos::Int,mps::Union{FiniteMPS,MPSComoving},cache::FinExEnv) = FinEx_AC2_eff(pos,mps,cache)


Base.:*(h::Union{<:FinEx_AC_eff,<:FinEx_AC2_eff},v) = h(v);

function (h::FinEx_AC_eff)(x)
    cache = h.envs;
    pos = h.pos;
    state = h.state;
    y = AC_eff(pos,state,cache.hamenv)*x

    for (ind,i) in enumerate(cache.overlaps)
        le = leftenv(i,pos,state);
        re = rightenv(i,pos,state);

        @plansor v[-1;-2 -3 -4] := le[4;-1 -2 5]*i.above.AC[pos][5 2;1]*re[1;-3 -4 3]*conj(x[4 2;3])
        @plansor y[-1 -2;-3] += conj(v[1;2 5 6])*(cache.weight*le)[-1;1 2 4]*i.above.AC[pos][4 -2;3]*re[3;5 6 -3]
    end

    y
end
function (h::FinEx_AC2_eff)(x)
    cache = h.envs;
    pos = h.pos;
    state = h.state;
    y = AC2_eff(pos,state,cache.hamenv)*x

    for (ind,i) in enumerate(cache.overlaps)
        le = leftenv(i,pos,state);
        re = rightenv(i,pos+1,state);


        @plansor v[-1;-2 -3 -4] := le[6;-1 -2 7]*i.above.AC[pos][7 4;5]*i.above.AR[pos+1][5 2;1]*re[1;-3 -4 3]*conj(x[6 4;3 2])
        @plansor y[-1 -2;-3 -4] += conj(v[2;3 5 6])*(cache.weight*le)[-1;2 3 4]*i.above.AC[pos][4 -2;7]*i.above.AR[pos+1][7 -4;1]*re[1;5 6 -3]
    end

    y
end
