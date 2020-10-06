"""
    mutable struct FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractMPS

Represents a finite matrix product state

When queried for AL/AR/AC/CL it will check if it is missing.
    If not, return
    If it is, calculate it, store it and return
"""
mutable struct FiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractMPS
    ALs::Vector{Union{Missing,A}}
    ARs::Vector{Union{Missing,A}}
    ACs::Vector{Union{Missing,A}}
    CLs::Vector{Union{Missing,B}}

    function FiniteMPS{A,B}(ALs::Vector{Union{Missing,A}},
                            ARs::Vector{Union{Missing,A}},
                            ACs::Vector{Union{Missing,A}},
                            CLs::Vector{Union{Missing,B}}) where {A<:GenericMPSTensor,B<:MPSBondTensor}
        # Relevant checks:
        # for every site at least one AL/AR/AC should not be missing
        # it should end and start with a trivial space
        # consecutive virtual spaces should 'fit' into eachother
        # checking the site tensor spaces
        # at least one CL/AC

        length(ACs) +1 == length(CLs) || throw(ArgumentError("length mismatch of AC/CL"))

        sum(ismissing.(ACs)) + sum(ismissing.(CLs)) < length(ACs)+length(CLs) || throw(ArgumentError("at least one AC/CL should not be missing"))

        prevD2 = nothing;

        for (i,tup) in enumerate(zip(ALs,ARs,ACs))
            non_missing = filter(!ismissing,tup)
            isempty(non_missing) && throw(ArgumentError("missing site tensor"))

            codom = codomain(non_missing[1])
            dom = domain(non_missing[1])
            for j in 2:length(non_missing)
                (dom == domain(non_missing[j]) && codom == codomain(non_missing[j])) || throw(SectorMismatch("AL/AC/AR should be maps over the same space"))
            end

            D1 = _firstspace(non_missing[1])
            D2 = _lastspace(non_missing[1])

            i == 1 || prevD2 == dual(D1) || throw(SectorMismatch("consecutive space tensors don't fit on the virtual level"))

            prevD2 = D2;
            ismissing(CLs[i]) || domain(CLs[i]) == codomain(CLs[i]) || throw(SectorMismatch("CL isn't a map between identical spaces"))
            ismissing(CLs[i+1]) || domain(CLs[i+1]) == codomain(CLs[i+1]) || throw(SectorMismatch("CL isn't a map between identical spaces"))
            ismissing(CLs[i])  || _lastspace(CLs[i]) == dual(D1) || throw(SectorMismatch("CL doesn't fit"))
            ismissing(CLs[i+1]) || _firstspace(CLs[i+1]) == dual(D2) || throw(SectorMismatch("CL doesn't fit"))

            #i != 1 || D1 == oneunit(D1) || throw(ArgumentError("finite mps should start with a trivial leg"))
            #i != length(ACs) || dual(D2) == oneunit(dual(D2)) || throw(ArgumentError("finite mps should end with a trivial leg"))
        end

        return new{A,B}(ALs,ARs,ACs,CLs);
    end
end

# allow construction with one large tensorkit space
function FiniteMPS(f, elt,P::ProductSpace, args...; kwargs...)
    return FiniteMPS(f, elt, collect(P), args...; kwargs...)
end

# allow construction given only a physical space and length
function FiniteMPS(f,elt, N::Int, V::VectorSpace, args...; kwargs...)
    return FiniteMPS(f, elt,fill(V, N), args...; kwargs...)
end

function FiniteMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S;
                    left::S = oneunit(maxvirtspace),
                    right::S = oneunit(maxvirtspace)) where {S<:ElementarySpace}
    N = length(physspaces)
    virtspaces = Vector{S}(undef, N+1)
    virtspaces[1] = left
    for k = 2:N
        virtspaces[k] = infimum(fuse(virtspaces[k-1], fuse(physspaces[k])), maxvirtspace)
    end
    virtspaces[N+1] = right
    for k = N:-1:2
        virtspaces[k] = infimum(virtspaces[k], fuse(virtspaces[k+1], flip(fuse(physspaces[k]))))
    end
    return FiniteMPS(f, elt,physspaces, virtspaces)
end
function FiniteMPS(f,elt,
                    physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                    virtspaces::Vector{S}) where {S<:ElementarySpace}
    N = length(physspaces)
    length(virtspaces) == N+1 || throw(DimensionMismatch())

    tensors = [TensorMap(f, elt,virtspaces[n] ⊗ physspaces[n], virtspaces[n+1]) for n=1:N]

    return FiniteMPS(tensors)
end


# allow construction with a simple array of tensors
function FiniteMPS(site_tensors::Vector{A}) where {A<:GenericMPSTensor}
    for i in 1:length(site_tensors)-1
        (site_tensors[i],C) = leftorth!(site_tensors[i],alg=QRpos());
        site_tensors[i+1] = _permute_front(C*_permute_tail(site_tensors[i+1]))
    end

    (site_tensors[end],C) = leftorth!(site_tensors[end],alg=QRpos());
    B = typeof(C);

    CLs = Vector{Union{Missing,B}}(missing,length(site_tensors)+1)
    ALs = Vector{Union{Missing,A}}(missing,length(site_tensors))
    ARs = Vector{Union{Missing,A}}(missing,length(site_tensors))
    ACs = Vector{Union{Missing,A}}(missing,length(site_tensors))

    ALs.= site_tensors;
    CLs[end] = C;

    FiniteMPS{A,B}(ALs,ARs,ACs,CLs)
end


Base.copy(psi::FiniteMPS{A,B}) where {A,B} = FiniteMPS{A,B}(copy(psi.ALs), copy(psi.ARs),copy(psi.ACs),copy(psi.CLs));

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
Base.eltype(st::FiniteMPS{A,B}) where {A<:GenericMPSTensor,B} = A
Base.eltype(::Type{FiniteMPS{A,B}}) where {A<:GenericMPSTensor,B} = A
bond_type(::Type{FiniteMPS{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Vtype

TensorKit.space(psi::FiniteMPS{<:MPSTensor}, n::Integer) = space(psi.AC[n], 2)
function TensorKit.space(psi::FiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = psi.AC[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

virtualspace(psi::FiniteMPS, n::Integer) =
    n < length(psi) ? _firstspace(psi.AC[n+1]) : dual(_lastspace(psi.AC[n]))

r_RR(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)},domain(state.AC[end]),domain(state.AC[end]))
l_LL(state::FiniteMPS{T}) where T = isomorphism(Matrix{eltype(T)},space(state.AC[1],1),space(state.AC[1],1))

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
    ρr = transfer_right(r_RR(psi2),psi2.AR[2:end],psi1.AR[2:end]);
    return tr(_permute_front(psi1.AC[1])' * _permute_front(psi2.AC[1]) * ρr)
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
