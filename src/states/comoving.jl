"
    MpsComoving(leftstate,window,rightstate)

    muteable window of tensors on top of an infinite chain
"
struct MpsComoving{Mtype<:GenMpsType,Vtype<:MpsVecType}
    left_gs::MpsCenterGauged{Mtype,Vtype}
    middle::Array{Mtype,1}
    right_gs::MpsCenterGauged{Mtype,Vtype}
end

Base.length(state::MpsComoving)=length(state.middle)
Base.getindex(state::MpsComoving,I...)=getindex(state.middle,I...)
Base.setindex!(state::MpsComoving,v,i)=setindex!(state.middle,v,i)
Base.copy(state::MpsComoving)=MpsComoving(state.left_gs,copy(state.middle),state.right_gs)
Base.deepcopy(state::MpsComoving)=MpsComoving(state.left_gs,deepcopy(state.middle),state.right_gs)
Base.lastindex(st::MpsComoving) = Base.length(st);

r_RR(state::MpsComoving)=r_RR(state.right_gs,length(state))
l_LL(state::MpsComoving)=l_LL(state.left_gs,1)

#we need the ability to copy the data from one mpscomoving into another mpscomoving
function Base.copyto!(st1::MpsComoving,st2::MpsComoving)
    for i in 1:length(st1)
        st1[i]=st2[i]
    end
    return st1
end

function expectation_value(state::Union{MpsComoving,FiniteMps},opp::TensorMap;leftorthed=false)
    if(!leftorthed)
        state=leftorth(state)
    end

    dat=[]

    for i in length(state):-1:1
        d=@tensor state[i][1,2,3]*opp[4,2]*conj(state[i][1,4,3])
        push!(dat,d)

        if i!=1
            (c,ar)=TensorKit.rightorth(state[i],(1,),(2,3))
            state[i]=permuteind(ar,(1,2),(3,))
            state[i-1]=state[i-1]*c
        end
    end

    return reverse(dat)
end
