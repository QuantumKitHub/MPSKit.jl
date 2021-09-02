# the transfer operation of a density matrix (with possible utility legs in its domain) by generic mps tensors - in a planar way!
function transfer_left(v::AbstractTensorMap{S,1,N2},A::GenericMPSTensor{S,N},Ab::GenericMPSTensor{S,N}=A) where {S,N2,N}
    t_v = transpose(v,reverse(ntuple(x->x,N2)),(N2+1,));
    t_A = _transpose_tail(A);
    adjoint(Ab)*transpose(t_v*t_A,(N2,reverse(ntuple(x->x+N2+1,N-1))...),(reverse(ntuple(x->x,N2-1))...,N2+1))
end
function transfer_right(v::AbstractTensorMap{S,1,N2},A::GenericMPSTensor{S,N},Ab::GenericMPSTensor{S,N}=A) where {S,N2,N}
    t_AV = transpose(A*v,(reverse(ntuple(x->N+x,N2-1))...,1),(N+N2,reverse(ntuple(x->x+1,N-1))...));
    t_Ab = _transpose_front(adjoint(Ab));
    transpose(t_AV*t_Ab,(N2,),(reverse(ntuple(x->x,N2-1))...,N2+1))
end

# the transfer operation of a density matrix with a utility leg in its codomain is ill defined - how should one braid the utility leg?
# hence the checks - to make sure that this operation is uniquely defined
function transfer_left(v::MPSTensor{S},A::MPSTensor{S},Ab::MPSTensor{S}=A) where S
    _can_unambiguously_braid(space(v,2)) || throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor v[-1 -2;-3] := v[1 2;4]*A[4 5;-3]*τ[2 3;5 -2]*conj(Ab[1 3;-1])
end
function transfer_right(v::MPSTensor{S},A::MPSTensor{S},Ab::MPSTensor{S}=A) where S
    _can_unambiguously_braid(space(v,2)) || throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor v[-1 -2;-3] := A[-1 2;1]*τ[-2 4;2 3]*conj(Ab[-3 4;5])*v[1 3;5]
end

# the transfer operation with a utility leg in both the domain and codomain is also ill defined - only due to the codomain utility space
function transfer_left(v::MPOTensor{S},A::MPSTensor{S},Ab::MPSTensor{S}=A) where S
    _can_unambiguously_braid(space(v,2)) || throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor t[-1 -2;-3 -4] := v[1 2;-3 4]*A[4 5;-4]*τ[2 3;5 -2]*conj(Ab[1 3;-1])
end
function transfer_right(v::MPOTensor{S},A::MPSTensor{S},Ab::MPSTensor{S}=A) where S
    _can_unambiguously_braid(space(v,2)) || throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor t[-1 -2;-3 -4] := A[-1 2;1]*τ[-2 4;2 3]*conj(Ab[-4 4;5])*v[1 3;-3 5]
end

#transfer for 2 mpo tensors
transfer_left(v::MPSBondTensor,A::MPOTensor,B::MPOTensor) =
    @plansor t[-1;-2] := v[1;2]*A[2 3;4 -2]*conj(B[1 3;4 -1])
transfer_right(v::MPSBondTensor,A::MPOTensor,B::MPOTensor) =
    @plansor t[-1;-2] := A[-1 3;4 1]*conj(B[-2 3;4 2])*v[1;2]

#mpo transfer
transfer_left(v::MPSTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) =
    @plansor v[-1 -2;-3] := v[1 2;4]*A[4 5;-3]*O[2 3;5 -2]*conj(Ab[1 3;-1])
transfer_right(v::MPSTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) =
    @plansor v[-1 -2;-3] := A[-1 2;1]*O[-2 4;2 3]*conj(Ab[-3 4;5])*v[1 3;5]

#utility, allowing transfering with arrays
function transfer_left(v,A::AbstractArray,Ab::AbstractArray=A;rvec=nothing,lvec=nothing)
    for (a,b) in zip(A,Ab)
        v = transfer_left(v,a,b)
    end

    if rvec != nothing && lvec != nothing
        if v isa MPSBondTensor #normal transfer
            @plansor v[-1;-2] -= rvec[1;2]*v[2;1]*lvec[-1;-2]
        elseif v isa MPSTensor #utiity leg in the middle
            @plansor v[-1 -2;-3] -= rvec[1;2]*v[2 -2;1]*lvec[-1;-3]
        else #what have you just given me?
            @assert false
        end
    end

    return v
end
function transfer_right(v,A::AbstractArray,Ab::AbstractArray=A;rvec=nothing,lvec=nothing)
    for (a,b) in Iterators.reverse(zip(A,Ab))
        v = transfer_right(v,a,b)
    end

    if rvec != nothing && lvec != nothing
        if v isa MPSBondTensor #normal transfer
            @plansor v[-1;-2]-=lvec[1;2]*v[2;1]*rvec[-1;-2]
        elseif v isa MPSTensor #utiity leg in the middle
            @plansor v[-1 -2;-3]-=lvec[1;2]*v[2 -2;1]*rvec[-1;-3]
        else
            @assert false
        end
    end

    return v
end
transfer_left(v,O::AbstractArray,A::AbstractArray,Ab::AbstractArray) =
    reduce((v,x)->transfer_left(v,x[1],x[2],x[3]),zip(O,A,Ab),init=v)
transfer_right(v,O::AbstractArray,A::AbstractArray,Ab::AbstractArray) =
    reduce((v,x)->transfer_right(v,x[1],x[2],x[3]),Iterators.reverse(zip(O,A,Ab)),init=v)
