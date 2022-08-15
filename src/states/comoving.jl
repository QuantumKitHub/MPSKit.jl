"
    MPSComoving(leftstate,window,rightstate)

    muteable window of tensors on top of an infinite chain
"
mutable struct MPSComoving{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS
    left_gs::InfiniteMPS{A,B}
    window::FiniteMPS{A,B}
    right_gs::InfiniteMPS{A,B}


    function MPSComoving(left::InfiniteMPS{A,B},window::FiniteMPS{A,B},right::InfiniteMPS{A,B}) where {A<:GenericMPSTensor,B<:MPSBondTensor}
        #todo:insert checks
        return new{A,B}(left,window,right);
    end
end

#allow construction with an array of tensors
MPSComoving(left::InfiniteMPS,site_tensors::Array,right::InfiniteMPS) = MPSComoving(left,FiniteMPS(site_tensors),right)

# allow construction with one large tensorkit space
MPSComoving(f, elt,P::ProductSpace, args...; kwargs...) = MPSComoving(f, elt, collect(P), args...; kwargs...)

# allow construction given only a physical space and length
MPSComoving(f,elt, N::Int, V::VectorSpace, args...; kwargs...) = MPSComoving(f, elt,fill(V, N), args...; kwargs...)

function MPSComoving(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S,
                    leftgs::M,
                    rightgs::M) where {S<:ElementarySpace,M<:InfiniteMPS}

    left = left_virtualspace(leftgs,0);
    right = right_virtualspace(rightgs,length(physspaces));
    window = FiniteMPS(f,elt,physspaces,maxvirtspace,left=left,right=right);
    MPSComoving(leftgs,window,rightgs)
end
function MPSComoving(f,elt,
                    physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                    virtspaces::Vector{S},
                    leftgs::M,
                    rightgs::M) where {S<:ElementarySpace,M<:InfiniteMPS}
    N = length(physspaces)
    length(virtspaces) == N+1 || throw(DimensionMismatch())

    tensors = [TensorMap(f, elt,virtspaces[n] ⊗ physspaces[n], virtspaces[n+1]) for n=1:N]

    return MPSComoving(leftgs,FiniteMPS(tensors,overwrite=true),rightgs)
end

#take a window from an infinite mps, and use that as state
function MPSComoving(state::InfiniteMPS{A,B},len::Int) where {A,B}
    CLs = Vector{Union{Missing,B}}(missing,len+1)
    ALs = Vector{Union{Missing,A}}(missing,len)
    ARs = Vector{Union{Missing,A}}(missing,len)
    ACs = Vector{Union{Missing,A}}(missing,len)

    ALs.= state.AL[1:len];
    ARs.= state.AR[1:len];
    ACs.= state.AC[1:len];
    CLs.= state.CR[0:len];

    MPSComoving(state,FiniteMPS(ALs,ARs,ACs,CLs),state)
end

Base.copy(state::MPSComoving{A,B}) where {A,B} = MPSComoving(state.left_gs,copy(state.window),state.right_gs);

Base.length(state::MPSComoving) = length(state.window)
Base.size(psi::MPSComoving, i...) = size(psi.window, i...)

Base.eltype(::Type{MPSComoving{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
site_type(::Type{MPSComoving{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
bond_type(::Type{MPSComoving{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Vtype


TensorKit.space(psi::MPSComoving{<:MPSTensor}, n::Integer) = space(psi.AC[n], 2)
left_virtualspace(psi::MPSComoving, n::Integer) = left_virtualspace(psi.window,n);
right_virtualspace(psi::MPSComoving, n::Integer) = right_virtualspace(psi.window,n);

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


max_Ds(f::MPSComoving) = max_Ds(f.window)

Base.:*(psi::MPSComoving, a::Number) = rmul!(copy(psi),a)
Base.:*(a::Number, psi::MPSComoving) = lmul!(a,copy(psi))

function TensorKit.lmul!(a::Number, psi::MPSComoving)
    psi.window.ACs .*=a;
    psi.window.CLs .*=a;
    return psi
end

function TensorKit.rmul!(psi::MPSComoving,a::Number)
    psi.window.ACs .*=a;
    psi.window.CLs .*=a;
    return psi
end


function TensorKit.dot(psi1::MPSComoving, psi2::MPSComoving)
    length(psi1) == length(psi2) || throw(ArgumentError("MPS with different length"))
    psi1.left_gs == psi2.left_gs || throw(ArgumentError("left InfiniteMPS is different"))
    psi1.right_gs == psi2.right_gs || throw(ArgumentError("right InfiniteMPS is different"))

    ρr = TransferMatrix(psi2.AR[2:end],psi1.AR[2:end])*r_RR(psi2);
    return tr(_transpose_front(psi1.AC[1])' * _transpose_front(psi2.AC[1]) * ρr)
end

TensorKit.norm(psi::MPSComoving) = norm(psi.AC[1])

TensorKit.normalize!(psi::MPSComoving) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::MPSComoving) = normalize!(copy(psi))
