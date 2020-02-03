"use a random matrix to expand ..."
@with_kw struct RandExpand<:Algorithm
    numvecs::Int = 1
end

#implemented for GenMpsType (wanted to use it in peps code)
function changebonds(state::Union{FiniteMps{T},MpsComoving{T}},alg::RandExpand) where T<:GenMpsType{Sp,N} where {Sp,N}
    lmax = zeros(length(state)+1)
    lmax[1] = dim(space(state[1],1))
    for i in 1:length(state)
        lmax[i+1]=dim(codomain(state[i]))*lmax[i]
    end

    rmax = ones(length(state)+1)
    rmax[end] = dim(space(state[length(state)],N+1))
    for i in length(state):-1:1
        rmax[i]=prod([dim(space(state[i],j)) for j in 2:N+1])*rmax[i+1]
    end

    minc = [minimum(z) for z in zip(lmax,rmax)][2:end]


    for i in 1:(length(state)-1)
        ninc = min(minc[i]-dim(space(state[i+1],N+1)),alg.numvecs) #the amount increased is the minimum of what is asked, and what is physically possible
        if ninc < 1
            continue
        end

        Vi=[TensorMap(rand,eltype(state[i+1]),oneunit(space(state[i+1],1)),prod([space(state[i+1],d)' for d in 2:N+1])) for j in 1:ninc]

        ar_re = reduce(TensorKit.catcodomain,Vi)

        state[i+1]=permuteind(TensorKit.catcodomain(permuteind(state[i+1],(1,),ntuple(x->x+1,Val{N}())),ar_re),ntuple(x->x,Val{N}()),(N+1,))

        ar_le=TensorMap(zeros,codomain(state[i]),space(ar_re,1))
        state[i]=TensorKit.catdomain(state[i],ar_le)

        (state[i],C)=TensorKit.leftorth!(state[i])
        state[i+1] = permuteind(C*permuteind(state[i+1],(1,),ntuple(x->x+1,Val{N}())),ntuple(x->x,Val{N}()),(N+1,))
        #@tensor state[i+1][-1 -2;-3] := C[-1,1]*state[i+1][1,-2,-3]
    end

    state = rightorth(state,renorm=false)
end
function changebonds(state::Union{FiniteMps,MpsComoving}, H::Hamiltonian,alg::RandExpand,pars=nothing)
    newstate = changebonds(state,alg);
    return newstate, (pars == nothing ? pars : params(newstate,H));
end

function changebonds(state::T, H::ComAct,alg,pars=nothing) where T <: FiniteMpo
    @info "$(typeof(alg)) not implemented for finite mpo; using slow fallback"

    mstate = rightorth!(mpo2mps.(state));
    (a,_) = splitham(H.below);
    (_,b) = splitham(H.above);
    nH = a+b
    (nmstate,_) = changebonds(mstate,nH,alg);

    nstate = rightorth!([mps2mpo(j,space(s,2)) for (j,s) in zip(nmstate,state)]::T)
    return nstate,params(nstate,H)
end
