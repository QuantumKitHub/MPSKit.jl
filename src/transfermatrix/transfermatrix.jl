abstract type AbstractTransferMatrix end;

# single site transfer
struct SingleTransferMatrix{A<:AbstractTensorMap,B,C<:AbstractTensorMap} <: AbstractTransferMatrix
    above::A
    middle::B
    below::C
    isflipped::Bool
end

#the product of transfer matrices is its own type
struct ProductTransferMatrix{T<:AbstractTransferMatrix} <: AbstractTransferMatrix
    tms :: Vector{T} # I don't want to use tuples, as an infinite mps transfer matrix will then be non-inferable
end

ProductTransferMatrix(v::AbstractVector) = ProductTransferMatrix(convert(Vector,v));

# a subset of possible operations, but certainly not all of them
Base.:*(prod::ProductTransferMatrix{T},tm::T) where T <:AbstractTransferMatrix = ProductTransferMatrix(vcat(prod.tms,tm));
Base.:*(tm::T,prod::ProductTransferMatrix{T}) where T <:AbstractTransferMatrix = ProductTransferMatrix(vcat(prod.tms,tm));
Base.:*(tm1::T,tm2::T) where T <: SingleTransferMatrix = ProductTransferMatrix([tm1,tm2]);

# regularized transfer matrices; where we project out after every full application
struct RegTransferMatrix{T<:AbstractTransferMatrix,L,R} <: AbstractTransferMatrix
    tm::T
    lvec::L
    rvec::R
end

#flip em
TensorKit.flip(tm::SingleTransferMatrix) = SingleTransferMatrix(tm.above,tm.middle,tm.below,!tm.isflipped);
TensorKit.flip(tm::ProductTransferMatrix) = ProductTransferMatrix(flip.(reverse(tm.tms)));
TensorKit.flip(tm::RegTransferMatrix) = RegTransferMatrix(flip(tm.tm),tm.rvec,tm.lvec);

# TransferMatrix acting on a vector using *
Base.:*(tm::AbstractTransferMatrix,vec) = tm(vec);
Base.:*(vec,tm::AbstractTransferMatrix) = flip(tm)(vec);

# TransferMatrix acting as a function
(d::ProductTransferMatrix)(vec) = foldr((a,b)->a(b),d.tms,init=vec);
(d::SingleTransferMatrix)(vec) = d.isflipped  ? transfer_left(vec,d.middle,d.above,d.below) : transfer_right(vec,d.middle,d.above,d.below);
(d::RegTransferMatrix)(vec) = regularize!(d.tm*vec,d.lvec,d.rvec);


# constructors
TransferMatrix(a) = TransferMatrix(a,nothing,a);
TransferMatrix(a,b) = TransferMatrix(a,nothing,b);
TransferMatrix(a::AbstractTensorMap,b,c::AbstractTensorMap,isflipped=false) = SingleTransferMatrix(a,b,c,isflipped);
function TransferMatrix(a::AbstractVector,b,c::AbstractVector,isflipped=false)
    tot = ProductTransferMatrix(convert(Vector,TransferMatrix.(a,b,c)));
    isflipped ? flip(tot) : tot
end


regularize(t::AbstractTransferMatrix,lvec,rvec) = RegTransferMatrix(t,lvec,rvec);

regularize!(v::MPSBondTensor,lvec::MPSBondTensor,rvec::MPSBondTensor) =
    @plansor v[-1;-2]-=lvec[1;2]*v[2;1]*rvec[-1;-2]

regularize!(v::MPSTensor,lvec::MPSBondTensor,rvec::MPSBondTensor) = 
    @plansor v[-1 -2;-3]-=lvec[1;2]*v[2 -2;1]*rvec[-1;-3]

regularize!(v::AbstractTensorMap{S,1,2} where S,lvec::MPSBondTensor,rvec::MPSBondTensor) =
    @plansor v[-1;-2 -3]-=lvec[1;2]*v[2;-2 1]*rvec[-1;-3]

regularize!(v::MPOTensor,lvec::MPSTensor,rvec::MPSTensor) =
    @plansor v[-1 -2;-3 -4] -= v[1 2;-3 3]*lvec[3 2;1]*rvec[-1 -2;-4]

regularize!(v::MPOTensor,lvec::MPSBondTensor,rvec::MPSBondTensor) =
    @plansor v[-1 -2;-3 -4] -= τ[6 2;3 4]*v[3 4;-3 5]*lvec[5;2]*rvec[-1;1]*τ[-2 -4;1 6]
