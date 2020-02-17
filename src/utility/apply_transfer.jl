#transfer
transfer_left(v::MPSVecType,A::M,Ab::M=A)  where M <: GenMPSType{S,N1} where {S,N1} = adjoint(Ab)*permute(v*permute(A,(1,),ntuple(x->x+1,Val{N1}())),ntuple(x->x,Val{N1}()),(N1+1,))
transfer_right(v::MPSVecType, A::M, Ab::M=A) where M <: GenMPSType{S,N1} where {S,N1} = permute(A*v,(1,),ntuple(x->x+1,Val{N1}()))*adjoint(permute(Ab,(1,),ntuple(x->x+1,Val{N1}())))

#transfer for 2 mpo tensors
transfer_left(v::MPSVecType,A::MPOType,B::MPOType) = @tensor t[-1;-2] := v[1,2]*A[2,3,-2,4]*conj(B[1,3,-1,4])
transfer_right(v::MPSVecType,A::MPOType,B::MPOType) = @tensor t[-1;-2] := A[-1,3,1,4]*conj(B[-2,3,2,4])*v[1,2]

#transfer, but there is a utility leg in the middle that is passed through
transfer_left(v::MPSType, A::MPSType, Ab::MPSType=A) = @tensor v[-1 -2;-3] := v[1,-2,2]*A[2,3,-3]*conj(Ab[1,3,-1])
transfer_right(v::MPSType, A::MPSType, Ab::MPSType=A) = @tensor v[-1 -2;-3] := A[-1,3,1]*v[1,-2,2]*conj(Ab[-3,3,2])

#mpo transfer
transfer_left(v::MPSType,O::MPOType,A::MPSType,Ab::MPSType) = @tensor v[-1 -2;-3] := v[4,2,1]*A[1,3,-3]*O[2,5,-2,3]*conj(Ab[4,5,-1])
transfer_right(v::MPSType,O::MPOType,A::MPSType,Ab::MPSType) = @tensor v[-1 -2;-3] := A[-1,4,5]*O[-2,2,3,4]*conj(Ab[-3,2,1])*v[5,3,1]

#utility, allowing transfering with arrays
function transfer_left(v,A::AbstractArray,Ab::AbstractArray=A;rvec=nothing,lvec=nothing)
    for (a,b) in zip(A,Ab)
        v = transfer_left(v,a,b)
    end

    if rvec != nothing && lvec != nothing
        if v isa MPSVecType #normal transfer
            @tensor v[-1;-2]-=rvec[1,2]*v[2,1]*lvec[-1,-2]
        elseif v isa MPSType #utiity leg in the middle
            @tensor v[-1 -2;-3]-=rvec[1,2]*v[2,-2,1]*lvec[-1,-3]
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
        if v isa MPSVecType #normal transfer
            @tensor v[-1;-2]-=lvec[1,2]*v[2,1]*rvec[-1,-2]
        elseif v isa MPSType #utiity leg in the middle
            @tensor v[-1 -2;-3]-=lvec[1,2]*v[2,-2,1]*rvec[-1,-3]
        else
            @assert false
        end
    end

    return v
end
transfer_left(v,O::AbstractArray,A::AbstractArray,Ab::AbstractArray) = reduce((v,x)->transfer_left(v,x[1],x[2],x[3]),zip(O,A,Ab),init=v)
transfer_right(v,O::AbstractArray,A::AbstractArray,Ab::AbstractArray) = reduce((v,x)->transfer_right(v,x[1],x[2],x[3]),Iterators.reverse(zip(O,A,Ab)),init=v)
