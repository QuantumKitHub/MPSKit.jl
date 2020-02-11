#we clearly should split of everything excitation related and give it a different name

#transfer
transfer_left(v::MpsVecType,A::M,Ab::M=A)  where M <: GenMpsType{S,N1} where {S,N1} = adjoint(Ab)*permute(v*permute(A,(1,),ntuple(x->x+1,Val{N1}())),ntuple(x->x,Val{N1}()),(N1+1,))
transfer_right(v::MpsVecType, A::M, Ab::M=A) where M <: GenMpsType{S,N1} where {S,N1} = permute(A*v,(1,),ntuple(x->x+1,Val{N1}()))*adjoint(permute(Ab,(1,),ntuple(x->x+1,Val{N1}())))

#transfer for 2 mpo tensors
transfer_left(v::MpsVecType,A::MpoType,B::MpoType) = @tensor t[-1;-2] := v[1,2]*A[2,3,-2,4]*conj(B[1,3,-1,4])
transfer_right(v::MpsVecType,A::MpoType,B::MpoType) = @tensor t[-1;-2] := A[-1,3,1,4]*conj(B[-2,3,2,4])*v[1,2]

#transfer, but the upper A is an excited tensor
transfer_left(v::MpsVecType, A::MpoType, Ab::MpsType) = @tensor t[-1 -2;-3] := v[1,2]*A[2,3,-2,-3]*conj(Ab[1,3,-1])
transfer_right(v::MpsVecType, A::MpoType, Ab::MpsType) = @tensor t[-1 -2;-3] := A[-1,3,-2,1]*v[1,2]*conj(Ab[-3,3,2])

#transfer, but there is a utility leg in the middle that is passed through
transfer_left(v::MpsType, A::MpsType, Ab::MpsType=A) = @tensor v[-1 -2;-3] := v[1,-2,2]*A[2,3,-3]*conj(Ab[1,3,-1])
transfer_right(v::MpsType, A::MpsType, Ab::MpsType=A) = @tensor v[-1 -2;-3] := A[-1,3,1]*v[1,-2,2]*conj(Ab[-3,3,2])

#transfer, but there is both a utility leg and an mpo leg that is passed through
transfer_left(v::MpoType, A::MpsType, Ab::MpsType=A) = @tensor v[-1 -2;-3 -4] := v[1,-2,-3,2]*A[2,3,-4]*conj(Ab[1,3,-1])
transfer_right(v::MpoType, A::MpsType, Ab::MpsType=A) = @tensor v[-1 -2;-3 -4] := A[-1,3,1]*v[1,-2,-3,2]*conj(Ab[-4,3,2])

#transfer, but the upper A is an excited tensor and there is an mpo leg being passed through
transfer_left(v::MpsType, A::MpoType, Ab::MpsType) = @tensor t[-1 -2;-3 -4] := v[1,-2,2]*A[2,3,-3,-4]*conj(Ab[1,3,-1])
transfer_right(v::MpsType, A::MpoType, Ab::MpsType) = @tensor t[-1 -2;-3 -4] := A[-1,3,-2,1]*v[1,-3,2]*conj(Ab[-4,3,2])

#mpo transfer
transfer_left(v::MpsType,O::MpoType,A::MpsType,Ab::MpsType) = @tensor v[-1 -2;-3] := v[4,2,1]*A[1,3,-3]*O[2,5,-2,3]*conj(Ab[4,5,-1])
transfer_right(v::MpsType,O::MpoType,A::MpsType,Ab::MpsType) = @tensor v[-1 -2;-3] := A[-1,4,5]*O[-2,2,3,4]*conj(Ab[-3,2,1])*v[5,3,1]

#mpo transfer, but with A an excitation-tensor
transfer_left(v::MpsType,O::MpoType,A::MpoType,Ab::MpsType) = @tensor t[-1 -2;-3 -4] := v[4,5,1]*A[1,3,-3,-4]*O[5,2,-2,3]*conj(Ab[4,2,-1])
transfer_right(v::MpsType,O::MpoType,A::MpoType,Ab::MpsType) = @tensor t[-1 -2;-3 -4] := A[-1,1,-2,5]*O[-3,3,4,1]*conj(Ab[-4,3,2])*v[5,4,2]

#mpo transfer, with an excitation leg
transfer_left(v::MpoType,O::MpoType,A::MpsType,Ab::MpsType=A) = @tensor v[-1 -2;-3 -4] := v[4,5,-3,1]*A[1,3,-4]*O[5,2,-2,3]*conj(Ab[4,2,-1])
transfer_right(v::MpoType,O::MpoType,A::MpsType,Ab::MpsType=A) = @tensor v[-1 -2;-3 -4] := A[-1,1,5]*O[-3,3,4,1]*conj(Ab[-4,3,2])*v[5,-2,4,2]

#utility, allowing transfering with arrays
function transfer_left(v,A::AbstractArray,Ab::AbstractArray=A;rvec=nothing,lvec=nothing)
    for (a,b) in zip(A,Ab)
        v = transfer_left(v,a,b)
    end

    if rvec != nothing && lvec != nothing
        if v isa MpsVecType #normal transfer
            @tensor v[-1;-2]-=rvec[1,2]*v[2,1]*lvec[-1,-2]
        elseif v isa MpsType #utiity leg in the middle
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
        if v isa MpsVecType #normal transfer
            @tensor v[-1;-2]-=lvec[1,2]*v[2,1]*rvec[-1,-2]
        elseif v isa MpsType #utiity leg in the middle
            @tensor v[-1 -2;-3]-=lvec[1,2]*v[2,-2,1]*rvec[-1,-3]
        else
            @assert false
        end
    end

    return v
end
transfer_left(v,O::AbstractArray,A::AbstractArray,Ab::AbstractArray) = reduce((v,x)->transfer_left(v,x[1],x[2],x[3]),zip(O,A,Ab),init=v)
transfer_right(v,O::AbstractArray,A::AbstractArray,Ab::AbstractArray) = reduce((v,x)->transfer_right(v,x[1],x[2],x[3]),Iterators.reverse(zip(O,A,Ab)),init=v)
