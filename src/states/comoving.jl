"
    MPSComoving(leftstate,window,rightstate)

    muteable window of tensors on top of an infinite chain
"
mutable struct MPSComoving{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractMPS
    left_gs::InfiniteMPS{A,B}

    ALs::Vector{Union{Missing,A}}
    ARs::Vector{Union{Missing,A}}
    ACs::Vector{Union{Missing,A}}
    CLs::Vector{Union{Missing,B}}

    right_gs::InfiniteMPS{A,B}


    function MPSComoving{A,B}(left,ALs::Vector{Union{Missing,A}},
                            ARs::Vector{Union{Missing,A}},
                            ACs::Vector{Union{Missing,A}},
                            CLs::Vector{Union{Missing,B}},right) where {A<:GenericMPSTensor,B<:MPSBondTensor}
        #todo:insert checks
        return new{A,B}(left,ALs,ARs,ACs,CLs,right);
    end
end

#allow construction with an array of tensors
function MPSComoving(left::InfiniteMPS{A,B},site_tensors::Array{A},right::InfiniteMPS{A,B}) where {A<:GenericMPSTensor,B<:MPSBondTensor}
    for i in 1:length(site_tensors)-1
        (site_tensors[i],C) = leftorth!(site_tensors[i],alg=QRpos());
        site_tensors[i+1] = _permute_front(C*_permute_tail(site_tensors[i+1]))
    end

    (site_tensors[end],C) = leftorth!(site_tensors[end],alg=QRpos());

    CLs = Vector{Union{Missing,B}}(missing,length(site_tensors)+1)
    ALs = Vector{Union{Missing,A}}(missing,length(site_tensors))
    ARs = Vector{Union{Missing,A}}(missing,length(site_tensors))
    ACs = Vector{Union{Missing,A}}(missing,length(site_tensors))

    ALs.= site_tensors;
    CLs[end] = C;

    MPSComoving{A,B}(left,ALs,ARs,ACs,CLs,right)
end

#todo : add constructor given an infinitemps + length

Base.copy(state::MPSComoving{A,B}) where {A,B} = MPSComoving{A,B}(state.left_gs,copy(state.ALs),copy(state.ARs),copy(state.ACs),copy(state.CLs),state.right_gs);

Base.length(state::MPSComoving) = length(state.ALs)
Base.size(psi::MPSComoving, i...) = size(psi.ALs, i...)

Base.eltype(::Type{MPSComoving{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
bond_type(::Type{MPSComoving{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Vtype

TensorKit.space(psi::MPSComoving{<:MPSTensor}, n::Integer) = space(psi.AC[n], 2)
virtualspace(psi::MPSComoving, n::Integer) =
    n < length(psi) ? _firstspace(psi.AC[n+1]) : dual(_lastspace(psi.AC[n]))


r_RR(state::MPSComoving)=r_RR(state.right_gs,length(state))
l_LL(state::MPSComoving)=l_LL(state.left_gs,1)

function Base.getproperty(psi::MPSComoving,prop::Symbol)
    if prop == :AL
        return ALView(psi)
    elseif prop == :AR
        return ARView(psi)
    elseif prop == :AC
        return ACView(psi)
    elseif prop == :CR
        return CRView(psi)
    else
        return getfield(psi,prop)
    end
end


function max_Ds(f::MPSComoving{G}) where G<:GenericMPSTensor{S,N} where {S,N}
    Ds = [dim(space(f.left_gs.AL[1],1)) for v in 1:length(f)+1];
    for i in 1:length(f)
        Ds[i+1] = Ds[i]*prod(map(x->dim(space(f.AC[i],x)),ntuple(x->x+1,Val{N-1}())))
    end

    Ds[end] = dim(space(f.right_gs.AL[1],1));
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(map(x->dim(space(f.AC[i],x)),ntuple(x->x+1,Val{N-1}()))))
    end
    Ds
end
function Base.:*(psi::MPSComoving, a::Number)
    return rmul!(copy(psi),a)
end

function Base.:*(a::Number, psi::MPSComoving)
    return lmul!(a,copy(psi))
end

function TensorKit.lmul!(a::Number, psi::MPSComoving)
    psi.ACs .*=a;
    psi.CLs .*=a;
    return psi
end

function TensorKit.rmul!(psi::MPSComoving,a::Number)
    psi.ACs .*=a;
    psi.CLs .*=a;
    return psi
end


function TensorKit.dot(psi1::MPSComoving, psi2::MPSComoving)
    length(psi1) == length(psi2) || throw(ArgumentError("MPS with different length"))
    psi1.left == psi2.left || throw(ArgumentError("left InfiniteMPS is different"))
    psi1.right == psi2.right || throw(ArgumentError("right InfiniteMPS is different"))

    return tr(_permute_front(psi1.AC[1])' * _permute_front(psi2.AC[1]))
end

TensorKit.norm(psi::MPSComoving) = norm(psi.AC[1])

TensorKit.normalize!(psi::MPSComoving) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::MPSComoving) = normalize!(copy(psi))
