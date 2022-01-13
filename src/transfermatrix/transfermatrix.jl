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

# a subset of possible operations, but certainly not all of them
Base.:*(prod::ProductTransferMatrix{T},tm::T) where T = ProductTransferMatrix(vcat(prod.tms,tm));
Base.:*(tm::T,prod::ProductTransferMatrix{T}) where T = ProductTransferMatrix(vcat(prod.tms,tm));
Base.:*(tm1::T,tm2::T) where T <: SingleTransferMatrix = ProductTransferMatrix([tm1,tm2]);

# regularized transfer matrices; where we project out after every full application
struct RegTransferMatrix{T<:AbstractTransferMatrix,L,R} <: AbstractTransferMatrix
    tm::T
    lvec::L
    rvec::R
end

#flip em
TensorKit.flip(tm::SingleTransferMatrix) = SingleTransferMatrix(tm.above,tm.middle,tm.below,tm.isflipped ⊻ true);
TensorKit.flip(tm::ProductTransferMatrix) = ProductTransferMatrix(flip.(reverse(tm.tms)));
TensorKit.flip(tm::RegTransferMatrix) = RegTransferMatrix(flip(tm.tm),tm.rvec,tm.lvec);

# TransferMatrix acting on a vector using *
Base.:*(tm::ProductTransferMatrix,vec) = foldr(*,tm.tms,init=vec);
Base.:*(vec,tm::ProductTransferMatrix) = foldl(*,tm.tms,init=vec);

Base.:*(tm::SingleTransferMatrix,vec) = tm(vec);
Base.:*(vec,tm::SingleTransferMatrix) = flip(tm)*vec;

Base.:*(tm::RegTransferMatrix,vec) = tm(vec);
Base.:*(vec,tm::RegTransferMatrix) = flip(tm)*vec;

# TransferMatrix acting as a function
(d::SingleTransferMatrix)(vec) = d.isflipped  ? transfer_left(vec,d.middle,d.above,d.below) : transfer_right(vec,d.middle,d.above,d.below);
function (d::RegTransferMatrix)(vec)
    v = d.tm*vec;

    if v isa MPSBondTensor #normal transfer
        @plansor v[-1;-2]-=d.lvec[1;2]*v[2;1]*d.rvec[-1;-2]
    elseif v isa MPSTensor #utiity leg in the middle
        @plansor v[-1 -2;-3]-=d.lvec[1;2]*v[2 -2;1]*d.rvec[-1;-3]
    else
        @assert false
    end

    v
end

# constructors
TransferMatrix(a;kwargs...) = TransferMatrix(a,nothing,a;kwargs...);
TransferMatrix(a,b;kwargs...) = TransferMatrix(a,nothing,b;kwargs...);
TransferMatrix(a,b,c;kwargs...) = TransferMatrix(a,b,c,false;kwargs...);
TransferMatrix(a::AbstractTensorMap,b,c::AbstractTensorMap,isflipped) = SingleTransferMatrix(a,b,c,isflipped);
function TransferMatrix(a::AbstractVector,b,c::AbstractVector,isflipped;lvec=nothing,rvec=nothing)
    tot = prod(TransferMatrix.(a,b,c));
    r_tot = isnothing(lvec) ? tot : RegTransferMatrix(tot,lvec,rvec);
    isflipped ? flip(r_tot) : r_tot
end
