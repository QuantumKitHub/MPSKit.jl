"""
    struct FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractMPS

Represents a finite matrix product state

When queried for AL/AR/AC/CL it will check if it is missing.
    If not, return
    If it is, calculate it, store it and return
"""
struct FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS
    ALs::Vector{Union{Missing,A}}
    ARs::Vector{Union{Missing,A}}
    ACs::Vector{Union{Missing,A}}
    CLs::Vector{Union{Missing,B}}

    function FiniteMPS(ALs::Vector{Union{Missing,A}},
                            ARs::Vector{Union{Missing,A}},
                            ACs::Vector{Union{Missing,A}},
                            CLs::Vector{Union{Missing,B}}) where {A<:GenericMPSTensor,B<:MPSBondTensor}

        length(ACs) +1 == length(CLs) || throw(ArgumentError("length mismatch of AC/CL"))

        sum(ismissing.(ACs)) + sum(ismissing.(CLs)) < length(ACs)+length(CLs) || throw(ArgumentError("at least one AC/CL should not be missing"))

        S = spacetype(A);
        left_virt_spaces = Vector{Union{Missing,S}}(missing,length(CLs));
        right_virt_spaces = Vector{Union{Missing,S}}(missing,length(CLs));

        for (i,tup) in enumerate(zip(ALs,ARs,ACs))
            non_missing = filter(!ismissing,tup)
            isempty(non_missing) && throw(ArgumentError("missing site tensor"))
            (al,ar,ac) = tup;

            if !ismissing(al)
                !ismissing(left_virt_spaces[i]) && (left_virt_spaces[i] == _firstspace(al) || throw(SectorMismatch("Virtual space of AL on site $(i) doesn't match")));

                left_virt_spaces[i+1] = _lastspace(al)';
                left_virt_spaces[i] = _firstspace(al)
            end

            if !ismissing(ar)
                !ismissing(right_virt_spaces[i]) && (right_virt_spaces[i] == _firstspace(ar) || throw(SectorMismatch("Virtual space of AR on site $(i) doesn't match")));

                right_virt_spaces[i+1] = _lastspace(ar)';
                right_virt_spaces[i] = _firstspace(ar)
            end

            if !ismissing(ac)
                !ismissing(left_virt_spaces[i]) && (left_virt_spaces[i] == _firstspace(ac) || throw(SectorMismatch("Left virtual space of AC on site $(i) doesn't match")));
                !ismissing(right_virt_spaces[i+1]) && (right_virt_spaces[i+1] == _lastspace(ac)' || throw(SectorMismatch("Right virtual space of AC on site $(i) doesn't match")));

                right_virt_spaces[i+1] = _lastspace(ac)';
                left_virt_spaces[i] = _firstspace(ac)
            end
        end

        for (i,c) in enumerate(CLs)
            ismissing(c) && continue;
            !ismissing(left_virt_spaces[i]) && (left_virt_spaces[i] == _firstspace(c) || throw(SectorMismatch("Left virtual space of CL on site $(i) doesn't match")));
            !ismissing(right_virt_spaces[i]) && (right_virt_spaces[i] == _lastspace(c)' || throw(SectorMismatch("Left virtual space of CL on site $(i) doesn't match")));
        end


        return new{A,B}(ALs,ARs,ACs,CLs);
    end
end

# allow construction with one large tensorkit space
FiniteMPS(P::ProductSpace,args...;kwargs...) = FiniteMPS(rand,Defaults.eltype,P,args...;kwargs...);
function FiniteMPS(f, elt,P::ProductSpace, args...; kwargs...)
    return FiniteMPS(f, elt, collect(P), args...; kwargs...)
end

# allow construction given only a physical space and length
FiniteMPS(N::Int,V::VectorSpace, args...;kwargs...)  = FiniteMPS(rand,Defaults.eltype,N,V,args...;kwargs...);
FiniteMPS(f,elt, N::Int, V::VectorSpace, args...; kwargs...) = FiniteMPS(f, elt,fill(V, N), args...; kwargs...)


FiniteMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S;kwargs...)  where {S<:ElementarySpace}= FiniteMPS(rand,Defaults.eltype,physspaces,maxvirtspace;kwargs...);
function FiniteMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S;
                    left::S = oneunit(maxvirtspace),
                    right::S = oneunit(maxvirtspace),kwargs...) where {S<:ElementarySpace}
    N = length(physspaces)
    virtspaces = Vector{S}(undef, N+1)
    virtspaces[1] = left
    for k = 2:N
        virtspaces[k] = infimum(fuse(virtspaces[k-1], fuse(physspaces[k])), maxvirtspace)
    end
    virtspaces[N+1] = right

    for k = N:-1:2
        virtspaces[k] = infimum(virtspaces[k], fuse(virtspaces[k+1], dual(fuse(physspaces[k]))))
    end

    return FiniteMPS(f, elt,physspaces, virtspaces;kwargs...)
end
function FiniteMPS(f,elt,
                    physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                    virtspaces::Vector{S};normalize=true) where {S<:ElementarySpace}
    N = length(physspaces)
    length(virtspaces) == N+1 || throw(DimensionMismatch())

    tensors = [TensorMap(f, elt,virtspaces[n] ⊗ physspaces[n], virtspaces[n+1]) for n=1:N]

    return FiniteMPS(tensors,normalize=normalize,overwrite=true)
end


# allow construction with a simple array of tensors
function FiniteMPS(site_tensors::Vector{A};normalize=false,overwrite=false) where {A<:GenericMPSTensor}
    site_tensors = overwrite ? site_tensors : copy(site_tensors);
    for i in 1:length(site_tensors)-1
        (site_tensors[i],C) = leftorth(site_tensors[i],alg=QRpos());
        normalize && normalize!(C);
        site_tensors[i+1] = _transpose_front(C*_transpose_tail(site_tensors[i+1]))
    end

    (site_tensors[end],C) = leftorth(site_tensors[end],alg=QRpos());
    normalize && normalize!(C);
    B = typeof(C);

    CLs = Vector{Union{Missing,B}}(missing,length(site_tensors)+1)
    ALs = Vector{Union{Missing,A}}(missing,length(site_tensors))
    ARs = Vector{Union{Missing,A}}(missing,length(site_tensors))
    ACs = Vector{Union{Missing,A}}(missing,length(site_tensors))

    ALs.= site_tensors;
    CLs[end] = C;

    FiniteMPS(ALs,ARs,ACs,CLs)
end

function Base.convert(TType::Type{<:AbstractTensorMap}, psi::FiniteMPS)
    T = foldl(psi.AR[2:end]; init=first(psi.AC)) do x, y
        return _transpose_front(x * _transpose_tail(y))
    end
    return convert(TType, T)
end

Base.copy(psi::FiniteMPS) = FiniteMPS(copy(psi.ALs), copy(psi.ARs),copy(psi.ACs),copy(psi.CLs));

function Base.getproperty(psi::FiniteMPS,prop::Symbol)
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

Base.length(psi::FiniteMPS) = length(psi.ALs)
Base.size(psi::FiniteMPS, i...) = size(psi.ALs, i...)

#conflicted if this is actually true
Base.eltype(st::FiniteMPS) = eltype(typeof(st));
Base.eltype(::Type{FiniteMPS{A,B}}) where {A<:GenericMPSTensor,B} = A

site_type(st::FiniteMPS) = site_type(typeof(st))
bond_type(st::FiniteMPS) = bond_type(typeof(st))
site_type(::Type{FiniteMPS{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
bond_type(::Type{FiniteMPS{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Vtype

TensorKit.space(psi::FiniteMPS{<:MPSTensor}, n::Integer) = space(psi.AC[n], 2)
function TensorKit.space(psi::FiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = psi.AC[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

left_virtualspace(psi::FiniteMPS, n::Integer) = _firstspace(psi.CR[n]);
right_virtualspace(psi::FiniteMPS, n::Integer) = dual(_lastspace(psi.CR[n]));

r_RR(state::FiniteMPS{T}) where T = isomorphism(storagetype(T),domain(state.AR[end]),domain(state.AR[end]))
l_LL(state::FiniteMPS{T}) where T = isomorphism(storagetype(T),space(state.AL[1],1),space(state.AL[1],1))

# Linear algebra methods
#------------------------

#=
    At this moment I don't yet make an effort to convert eltype(finitemps) to make it multiply-able by a
    also don't use lmul! from tensorkit, so this thing copies
=#
function Base.:*(psi::FiniteMPS, a::Number)
    return rmul!(copy(psi),a)
end

function Base.:*(a::Number, psi::FiniteMPS)
    return lmul!(a,copy(psi))
end

function TensorKit.lmul!(a::Number, psi::FiniteMPS)
    psi.ACs .*=a;
    psi.CLs .*=a;
    return psi
end

function TensorKit.rmul!(psi::FiniteMPS,a::Number)
    psi.ACs .*=a;
    psi.CLs .*=a;
    return psi
end

function TensorKit.dot(psi1::FiniteMPS, psi2::FiniteMPS)
    #todo : rewrite this without having to gauge
    length(psi1) == length(psi2) || throw(ArgumentError("MPS with different length"))
    ρr = TransferMatrix(psi2.AR[2:end],psi1.AR[2:end])*r_RR(psi2)
    return tr(_transpose_front(psi1.AC[1])' * _transpose_front(psi2.AC[1]) * ρr)
end

#todo : rewrite this without having to gauge
TensorKit.norm(psi::FiniteMPS) = norm(psi.AC[1])
TensorKit.normalize!(psi::FiniteMPS) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::FiniteMPS) = normalize!(copy(psi))


"
    A FiniteMPS/MPO starts and ends with a bond dimension 1 leg
    Its bond dimension also can't grow faster then pspaces (resp. pspaces^2)
    This function calculates the maximal achievable bond dimension
"
function max_Ds(f::FiniteMPS{G}) where G<:GenericMPSTensor{S,N} where {S,N}
    Ds = [1 for v in 1:length(f)+1];
    for i in 1:length(f)
        Ds[i+1] = Ds[i]*prod(map(x->dim(space(f.AC[i],x)),ntuple(x->x+1,Val{N-1}())))
    end

    Ds[end] = 1;
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(map(x->dim(space(f.AC[i],x)),ntuple(x->x+1,Val{N-1}()))))
    end
    Ds
end

Base.:-(psi1::FiniteMPS,psi2::FiniteMPS) = psi1+(-1*psi2);
function Base.:+(psi1::FiniteMPS{A}, psi2::FiniteMPS{A}) where A
    length(psi1) == length(psi2) || throw(DimensionMismatch())
    N = length(psi1)
    for k = 1:N
        space(psi1, k) == space(psi2, k) || throw(SpaceMismatch("Non-matching physical space on site $k."))
    end
    left_virtualspace(psi1, 0) == left_virtualspace(psi2, 0) || throw(SpaceMismatch("Non-matching left virtual space."))
    right_virtualspace(psi1, N) == right_virtualspace(psi2, N) || throw(SpaceMismatch("Non-matching right virtual space."))

    tensors = A[]

    k = 1 # firstindex(psi1)
    t1 = psi1.AL[k]
    t2 = psi2.AL[k]
    V1 = domain(t1)[1]
    V2 = domain(t2)[1]
    w1 = isometry(storagetype(A), V1 ⊕ V2, V1)
    w2 = leftnull(w1)
    @assert domain(w2) == ⊗(V2)

    push!(tensors,t1*w1' + t2*w2')
    for k = 2:N-1
        t1 = _transpose_front(w1*_transpose_tail(psi1.AL[k]))
        t2 = _transpose_front(w2*_transpose_tail(psi2.AL[k]))
        V1 = domain(t1)[1]
        V2 = domain(t2)[1]
        w1 = isometry(storagetype(A), V1 ⊕ V2, V1)
        w2 = leftnull(w1)
        @assert domain(w2) == ⊗(V2)
        push!(tensors, t1*w1' + t2*w2')
    end
    k = N
    t1 = _transpose_front(w1*_transpose_tail(psi1.AC[k]))
    t2 = _transpose_front(w2*_transpose_tail(psi2.AC[k]))
    push!(tensors, t1 + t2)
    return FiniteMPS(tensors)
end
