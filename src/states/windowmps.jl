"""
    WindowMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS

Type that represents a finite Matrix Product State embedded in an infinte Matrix Product State.

## Fields

- `left_gs::InfiniteMPS` -- left infinite environment
- `window::FiniteMPS` -- finite window Matrix Product State
- `right_gs::InfiniteMPS` -- right infinite environment

---

## Constructors

    WindowMPS(left_gs::InfiniteMPS, window_state::FiniteMPS, [right_gs::InfiniteMPS])
    WindowMPS(left_gs::InfiniteMPS, window_tensors::AbstractVector, [right_gs::InfiniteMPS])
    WindowMPS([f, eltype], physicalspaces::Vector{<:Union{S, CompositeSpace{S}},
              virtualspaces::Vector{<:Union{S, CompositeSpace{S}}, left_gs::InfiniteMPS,
              [right_gs::InfiniteMPS])
    WindowMPS([f, eltype], physicalspaces::Vector{<:Union{S,CompositeSpace{S}}},
              maxvirtualspace::S, left_gs::InfiniteMPS, [right_gs::InfiniteMPS])
    
Construct a WindowMPS via a specification of left and right infinite environment, and either
a window state or a vector of tensors to construct the window. Alternatively, it is possible
to supply the same arguments as for the constructor of [`FiniteMPS`](@ref), followed by a
left (and right) environment to construct the WindowMPS in one step.

    WindowMPS(state::InfiniteMPS, L::Int)

Construct a WindowMPS from an InfiniteMPS, by promoting a region of length `L` to a
`FiniteMPS`.
"""
mutable struct WindowMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS
    left_gs::InfiniteMPS{A,B}
    window::FiniteMPS{A,B}
    right_gs::InfiniteMPS{A,B}


    function WindowMPS(left::InfiniteMPS{A,B},window::FiniteMPS{A,B},right::InfiniteMPS{A,B}) where {A<:GenericMPSTensor,B<:MPSBondTensor}
        #todo:insert checks
        return new{A,B}(left,window,right);
    end
end

#allow construction with an array of tensors
WindowMPS(left::InfiniteMPS,site_tensors::Array,right::InfiniteMPS) = WindowMPS(left,FiniteMPS(site_tensors),right)

# allow construction with one large tensorkit space
WindowMPS(f, elt,P::ProductSpace, args...; kwargs...) = WindowMPS(f, elt, collect(P), args...; kwargs...)

# allow construction given only a physical space and length
WindowMPS(f,elt, N::Int, V::VectorSpace, args...; kwargs...) = WindowMPS(f, elt,fill(V, N), args...; kwargs...)

function WindowMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S,
                    leftgs::M,
                    rightgs::M) where {S<:ElementarySpace,M<:InfiniteMPS}

    left = left_virtualspace(leftgs,0);
    right = right_virtualspace(rightgs,length(physspaces));
    window = FiniteMPS(f,elt,physspaces,maxvirtspace,left=left,right=right);
    WindowMPS(leftgs,window,rightgs)
end
function WindowMPS(f,elt,
                    physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                    virtspaces::Vector{S},
                    leftgs::M,
                    rightgs::M) where {S<:ElementarySpace,M<:InfiniteMPS}
    N = length(physspaces)
    length(virtspaces) == N+1 || throw(DimensionMismatch())

    tensors = [TensorMap(f, elt,virtspaces[n] ⊗ physspaces[n], virtspaces[n+1]) for n=1:N]

    return WindowMPS(leftgs,FiniteMPS(tensors,overwrite=true),rightgs)
end

#take a window from an infinite mps, and use that as state
function WindowMPS(state::InfiniteMPS{A,B},len::Int) where {A,B}
    CLs = Vector{Union{Missing,B}}(missing,len+1)
    ALs = Vector{Union{Missing,A}}(missing,len)
    ARs = Vector{Union{Missing,A}}(missing,len)
    ACs = Vector{Union{Missing,A}}(missing,len)

    ALs.= state.AL[1:len];
    ARs.= state.AR[1:len];
    ACs.= state.AC[1:len];
    CLs.= state.CR[0:len];

    WindowMPS(state,FiniteMPS(ALs,ARs,ACs,CLs),state)
end

Base.copy(state::WindowMPS{A,B}) where {A,B} = WindowMPS(state.left_gs,copy(state.window),state.right_gs);

Base.length(state::WindowMPS) = length(state.window)
Base.size(psi::WindowMPS, i...) = size(psi.window, i...)

Base.eltype(::Type{WindowMPS{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
site_type(::Type{WindowMPS{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
bond_type(::Type{WindowMPS{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Vtype
site_type(st::WindowMPS) = site_type(typeof(st))
bond_type(st::WindowMPS) = bond_type(typeof(st))

TensorKit.space(psi::WindowMPS{<:MPSTensor}, n::Integer) = space(psi.AC[n], 2)
left_virtualspace(psi::WindowMPS, n::Integer) = left_virtualspace(psi.window,n);
right_virtualspace(psi::WindowMPS, n::Integer) = right_virtualspace(psi.window,n);

r_RR(state::WindowMPS)=r_RR(state.right_gs,length(state))
l_LL(state::WindowMPS)=l_LL(state.left_gs,1)

function Base.getproperty(psi::WindowMPS,prop::Symbol)
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


max_Ds(f::WindowMPS) = max_Ds(f.window)

Base.:*(psi::WindowMPS, a::Number) = rmul!(copy(psi),a)
Base.:*(a::Number, psi::WindowMPS) = lmul!(a,copy(psi))

function TensorKit.lmul!(a::Number, psi::WindowMPS)
    psi.window.ACs .*=a;
    psi.window.CLs .*=a;
    return psi
end

function TensorKit.rmul!(psi::WindowMPS,a::Number)
    psi.window.ACs .*=a;
    psi.window.CLs .*=a;
    return psi
end


function TensorKit.dot(psi1::WindowMPS, psi2::WindowMPS)
    length(psi1) == length(psi2) || throw(ArgumentError("MPS with different length"))
    psi1.left_gs == psi2.left_gs || throw(ArgumentError("left InfiniteMPS is different"))
    psi1.right_gs == psi2.right_gs || throw(ArgumentError("right InfiniteMPS is different"))

    ρr = TransferMatrix(psi2.AR[2:end],psi1.AR[2:end])*r_RR(psi2);
    return tr(_transpose_front(psi1.AC[1])' * _transpose_front(psi2.AC[1]) * ρr)
end

TensorKit.norm(psi::WindowMPS) = norm(psi.AC[1])

TensorKit.normalize!(psi::WindowMPS) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::WindowMPS) = normalize!(copy(psi))
