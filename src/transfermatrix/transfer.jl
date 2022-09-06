# ------------------------------------------
# | transfers for (vector, tensor, tensor) |
# ------------------------------------------

# transfer of density matrix (with possible utility legs in its domain) by generic mps tensors
function transfer_left(v::AbstractTensorMap{S,1,N2},A::GenericMPSTensor{S,N},Ab::GenericMPSTensor{S,N}) where {S,N2,N}
    t_v = transpose(v,reverse(ntuple(x->x,N2)),(N2+1,));
    t_A = _transpose_tail(A);
    adjoint(Ab)*transpose(t_v*t_A,(N2,reverse(ntuple(x->x+N2+1,N-1))...),(reverse(ntuple(x->x,N2-1))...,N2+1))
end
function transfer_right(v::AbstractTensorMap{S,1,N2},A::GenericMPSTensor{S,N},Ab::GenericMPSTensor{S,N}) where {S,N2,N}
    t_AV = transpose(A*v,(reverse(ntuple(x->N+x,N2-1))...,1),(N+N2,reverse(ntuple(x->x+1,N-1))...));
    t_Ab = _transpose_front(adjoint(Ab));
    transpose(t_AV*t_Ab,(N2,),(reverse(ntuple(x->x,N2-1))...,N2+1))
end

#transfer, but the upper A is an excited tensor
transfer_left(v::MPSBondTensor, A::MPOTensor, Ab::MPSTensor) =
    @plansor t[-1;-2 -3] := v[1;2]*A[2 3;-2 -3]*conj(Ab[1 3;-1])
transfer_right(v::MPSBondTensor, A::MPOTensor, Ab::MPSTensor) =
    @plansor t[-1;-2 -3] := A[-1 3;-2 1]*v[1;2]*conj(Ab[-3 3;2])

#transfer, but the upper A is an excited tensor and there is an mpo leg being passed through
transfer_left(v::MPSTensor, A::MPOTensor, Ab::MPSTensor) =
    @plansor t[-1 -2;-3 -4] := v[1 3;4]*A[4 5;-3 -4]*τ[3 2;5 -2]*conj(Ab[1 2;-1])

transfer_right(v::MPSTensor, A::MPOTensor, Ab::MPSTensor) =
    @plansor t[-1 -2;-3 -4] := A[-1 4;-3 5]*τ[-2 3;4 2]*conj(Ab[-4 3;1])*v[5 2;1]


# the transfer operation of a density matrix with a utility leg in its codomain is ill defined - how should one braid the utility leg?
# hence the checks - to make sure that this operation is uniquely defined
function transfer_left(v::MPSTensor{S},A::MPSTensor{S},Ab::MPSTensor{S}) where S
    _can_unambiguously_braid(space(v,2)) || throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor v[-1 -2;-3] := v[1 2;4]*A[4 5;-3]*τ[2 3;5 -2]*conj(Ab[1 3;-1])
end
function transfer_right(v::MPSTensor{S},A::MPSTensor{S},Ab::MPSTensor{S}) where S
    _can_unambiguously_braid(space(v,2)) || throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor v[-1 -2;-3] := A[-1 2;1]*τ[-2 4;2 3]*conj(Ab[-3 4;5])*v[1 3;5]
end

# the transfer operation with a utility leg in both the domain and codomain is also ill defined - only due to the codomain utility space
function transfer_left(v::MPOTensor{S},A::MPSTensor{S},Ab::MPSTensor{S}) where S
    _can_unambiguously_braid(space(v,2)) || throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor t[-1 -2;-3 -4] := v[1 2;-3 4]*A[4 5;-4]*τ[2 3;5 -2]*conj(Ab[1 3;-1])
end
function transfer_right(v::MPOTensor{S},A::MPSTensor{S},Ab::MPSTensor{S}) where S
    _can_unambiguously_braid(space(v,2)) || throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor t[-1 -2;-3 -4] := A[-1 2;1]*τ[-2 4;2 3]*conj(Ab[-4 4;5])*v[1 3;-3 5]
end

#transfer for 2 mpo tensors
transfer_left(v::MPSBondTensor,A::MPOTensor,B::MPOTensor) =
    @plansor t[-1;-2] := v[1;2]*A[2 3;4 -2]*conj(B[1 3;4 -1])
transfer_right(v::MPSBondTensor,A::MPOTensor,B::MPOTensor) =
    @plansor t[-1;-2] := A[-1 3;4 1]*conj(B[-2 3;4 2])*v[1;2]

# ----------------------------------------------------
# | transfers for (vector, operator, tensor, tensor) |
# ----------------------------------------------------

transfer_left(v,::Nothing,A,B) = transfer_left(v,A,B);
transfer_right(v,::Nothing,A,B) = transfer_right(v,A,B);

#mpo transfer
transfer_left(v::MPSTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) =
    @plansor v[-1 -2;-3] := v[1 2;4]*A[4 5;-3]*O[2 3;5 -2]*conj(Ab[1 3;-1])
transfer_right(v::MPSTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) =
    @plansor v[-1 -2;-3] := A[-1 2;1]*O[-2 4;2 3]*conj(Ab[-3 4;5])*v[1 3;5]


#mpo transfer, but with A an excitation-tensor
transfer_left(v::MPSTensor,O::MPOTensor,A::MPOTensor,Ab::MPSTensor) =
    @plansor t[-1 -2;-3 -4] := v[4 2;1]*A[1 3;-3 -4]*O[2 5;3 -2]*conj(Ab[4 5;-1])
transfer_right(v::MPSTensor,O::MPOTensor,A::MPOTensor,Ab::MPSTensor) =
    @plansor t[-1 -2;-3 -4] := A[-1 4;-3 5]*O[-2 2;4 3]*conj(Ab[-4 2;1])*v[5 3;1]

#mpo transfer, with an excitation leg
transfer_left(v::MPOTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) =
    @plansor v[-1 -2;-3 -4] := v[4 2;-3 1]*A[1 3;-4]*O[2 5;3 -2]*conj(Ab[4 5;-1])
transfer_right(v::MPOTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) =
    @plansor v[-1 -2;-3 -4] := A[-1 4;5]*O[-2 2;4 3]*conj(Ab[-4 2;1])*v[5 3;-3 1]

# --- the following really needs a proper rewrite; probably without transducers
transfer_left(vec::RecursiveVec,opp,A,Ab) = RecursiveVec(transfer_left(vec.vecs,opp,A,Ab));
transfer_right(vec::RecursiveVec,opp,A,Ab) = RecursiveVec(transfer_right(vec.vecs,opp,A,Ab));


# usual sparsemposlice transfer
transfer_left(vec::AbstractVector{V},ham::SparseMPOSlice,A::V,Ab::V) where V<:MPSTensor =
    transfer_left(V,vec,ham,A,Ab)
transfer_right(vec::AbstractVector{V},ham::SparseMPOSlice,A::V,Ab::V) where V<:MPSTensor =
    transfer_right(V,vec,ham,A,Ab)

# A excited
transfer_left(vec::AbstractVector{V},ham::SparseMPOSlice,A::M,Ab::V) where V<:MPSTensor where M <:MPOTensor =
    transfer_left(M,vec,ham,A,Ab)
transfer_right(vec::AbstractVector{V},ham::SparseMPOSlice,A::M,Ab::V) where V<:MPSTensor where M <:MPOTensor =
    transfer_right(M,vec,ham,A,Ab)

# vec excited
transfer_left(vec::AbstractVector{V},ham::SparseMPOSlice,A::M,Ab::M) where V<:MPOTensor where M <:MPSTensor =
    transfer_left(V,vec,ham,A,Ab)
transfer_right(vec::AbstractVector{V},ham::SparseMPOSlice,A::M,Ab::M) where V<:MPOTensor where M <:MPSTensor =
    transfer_right(V,vec,ham,A,Ab)

function transfer_left(RetType,vec,ham::SparseMPOSlice,A,Ab)
    toret = similar(vec,RetType,length(vec));

    @threads for k in 1:length(vec)

        els = keys(ham,:,k);


        @floop WorkStealingEx() for j in els
            if isscal(ham,j,k)
                t = lmul!(ham.Os[j,k], transfer_left(vec[j],A,Ab))
            else
                t = transfer_left(vec[j],ham[j,k],A,Ab)
            end

            @reduce(s = inplace_add!(nothing,t))
        end

        if isnothing(s)
            s = transfer_left(vec[1],ham[1,k],A,Ab)
        end
        toret[k] = s;
    end

    toret
end
function transfer_right(RetType,vec,ham::SparseMPOSlice,A,Ab)
    toret = similar(vec,RetType,length(vec));

    @threads for j in 1:length(vec)

        els = keys(ham,j,:)

        @floop WorkStealingEx() for k in els
            if isscal(ham,j,k)
                t = lmul!(ham.Os[j,k],transfer_right(vec[k],A,Ab))
            else 
                t = transfer_right(vec[k],ham[j,k],A,Ab)
            end

            @reduce(s = inplace_add!(nothing,t))
        end

        if isnothing(s)
            s = transfer_right(vec[1],ham[j,1],A,Ab)
        end

        toret[j] = s
    end

    toret
end

