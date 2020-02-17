"
    MPSComoving(leftstate,window,rightstate)

    muteable window of tensors on top of an infinite chain
"
struct MPSComoving{Mtype<:GenMPSType,Vtype<:MPSVecType}
    left_gs::InfiniteMPS{Mtype,Vtype}
    middle::Array{Mtype,1}
    right_gs::InfiniteMPS{Mtype,Vtype}
end

Base.length(state::MPSComoving)=length(state.middle)
Base.getindex(state::MPSComoving,I...)=getindex(state.middle,I...)
Base.setindex!(state::MPSComoving,v,i)=setindex!(state.middle,v,i)
Base.copy(state::MPSComoving)=MPSComoving(state.left_gs,copy(state.middle),state.right_gs)
Base.deepcopy(state::MPSComoving)=MPSComoving(state.left_gs,deepcopy(state.middle),state.right_gs)
Base.lastindex(st::MPSComoving) = Base.length(st);

r_RR(state::MPSComoving)=r_RR(state.right_gs,length(state))
l_LL(state::MPSComoving)=l_LL(state.left_gs,1)

#we need the ability to copy the data from one mpscomoving into another mpscomoving
function Base.copyto!(st1::MPSComoving,st2::MPSComoving)
    for i in 1:length(st1)
        st1[i]=st2[i]
    end
    return st1
end

function expectation_value(state::Union{MPSComoving,FiniteMPS},opp::TensorMap;leftorthed=false)
    if(!leftorthed)
        state=leftorth(state)
    end

    dat=[]

    for i in length(state):-1:1
        d=@tensor state[i][1,2,3]*opp[4,2]*conj(state[i][1,4,3])
        push!(dat,d)

        if i!=1
            (c,ar)=TensorKit.rightorth(state[i],(1,),(2,3))
            state[i]=permute(ar,(1,2),(3,))
            state[i-1]=state[i-1]*c
        end
    end

    return reverse(dat)
end
