"
    MPSComoving(leftstate,window,rightstate)

    muteable window of tensors on top of an infinite chain
"
mutable struct MPSComoving{Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} <: AbstractMPS
    left_gs::InfiniteMPS{Mtype,Vtype}

    site_tensors::Vector{Mtype}
    bond_tensors::Vector{Union{Missing,Vtype}}

    right_gs::InfiniteMPS{Mtype,Vtype}

    gaugedpos::Tuple{Int,Int} # range of tensors which are not left or right normalized

    function MPSComoving{Mtype,Vtype}(left::InfiniteMPS{Mtype,Vtype},site_tensors::Array{Mtype,1},right::InfiniteMPS{Mtype,Vtype},bond_tensors::Vector{Union{Missing,Vtype}},gaugedpos::Tuple{Int,Int}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor}
        #todo:insert checks
        return new{Mtype,Vtype}(left, site_tensors, bond_tensors, right, gaugedpos)
    end
end

#allow construction with only site_tensors
MPSComoving(left::InfiniteMPS{Mtype,Vtype},site_tensors::Array{Mtype,1},right::InfiniteMPS{Mtype,Vtype},
            bond_tensors::Vector{Union{Missing,Vtype}} = Vector{Union{Missing,Vtype}}(missing,length(site_tensors)+1),
            gaugedpos::Tuple{Int,Int} = (0,length(site_tensors)+1)) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = MPSComoving{Mtype,Vtype}(left,site_tensors,right,bond_tensors,gaugedpos)

Base.copy(state::MPSComoving) = MPSComoving(state.left_gs,deepcopy(state.site_tensors),state.right_gs,deepcopy(state.bond_tensors),state.gaugedpos)
Base.length(state::MPSComoving) = length(state.site_tensors)
Base.size(psi::MPSComoving, i...) = size(psi.site_tensors, i...)
Base.eltype(::Type{MPSComoving{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
Base.similar(psi::MPSComoving) = MPSComoving(psi.left_gs,similar.(psi.site_tensors),psi.right_gs)

TensorKit.space(psi::MPSComoving{<:MPSTensor}, n::Integer) = space(psi.site_tensors[n], 2)
virtualspace(psi::MPSComoving, n::Integer) =
    n < length(psi) ? _firstspace(psi.site_tensors[n+1]) : dual(_lastspace(psi.site_tensors[n]))


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
        Ds[i+1] = Ds[i]*prod(map(x->dim(space(f.site_tensors[i],x)),ntuple(x->x+1,Val{N-1}())))
    end

    Ds[end] = dim(space(f.right_gs.AL[1],1));
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(map(x->dim(space(f.site_tensors[i],x)),ntuple(x->x+1,Val{N-1}()))))
    end
    Ds
end

"""
    function leftorth!(psi::MPSComoving, n = length(psi); normalize = true, alg = QRpos())

Bring all MPS tensors of `psi` up to and including site `n` into left orthonormal form, using
the orthogonal factorization algorithm `alg`
"""
function TensorKit.leftorth!(psi::MPSComoving, n::Integer = length(psi);
                    alg::OrthogonalFactorizationAlgorithm = QRpos(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)

    while first(psi.gaugedpos) < n
        k = first(psi.gaugedpos) + 1

        if !ismissing(psi.bond_tensors[k])
            C = psi.bond_tensors[k];
            psi.bond_tensors[k] = missing;
            psi.site_tensors[k] = _permute_front(C*_permute_tail(psi.site_tensors[k]))
        end

        psi.site_tensors[k], psi.bond_tensors[k+1] = leftorth(psi.site_tensors[k]; alg = alg)
        psi.gaugedpos = (k,max(k+1,last(psi.gaugedpos)))
    end
    return normalize ? normalize!(psi) : psi
end
function TensorKit.rightorth!(psi::MPSComoving, n::Integer = 1;
                    alg::OrthogonalFactorizationAlgorithm = LQpos(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)

    while last(psi.gaugedpos) > n
        k = last(psi.gaugedpos) - 1

        if !ismissing(psi.bond_tensors[k+1])
            C = psi.bond_tensors[k+1];
            psi.bond_tensors[k+1] = missing;
            psi.site_tensors[k] = psi.site_tensors[k]*C
        end

        C, AR = rightorth(_permute_tail(psi.site_tensors[k]); alg = alg)

        psi.site_tensors[k] = _permute_front(AR)
        psi.bond_tensors[k] = C;

        psi.gaugedpos = (min(k-1,first(psi.gaugedpos)), k)
    end

    return normalize ? normalize!(psi) : psi
end


function Base.:*(psi::MPSComoving, a::Number)
    nsite_tensors = psi.site_tensors .* one(a)
    nbond_tensors = convert(Vector{Union{Missing,bond_type(nsite_tensors[1])}}, psi.bond_tensors .* one(a))

    psi′ = MPSComoving(psi.left_gs,nsite_tensors ,psi.right_gs, nbond_tensors,psi.gaugedpos)
    return rmul!(psi′, a)
end

function Base.:*(a::Number, psi::MPSComoving)
    nsite_tensors = psi.site_tensors .* one(a)
    nbond_tensors = convert(Vector{Union{Missing,bond_type(nsite_tensors[1])}}, psi.bond_tensors .* one(a))

    psi′ = MPSComoving(psi.left_gs,nsite_tensors ,psi.right_gs, nbond_tensors,psi.gaugedpos)
    return lmul!(a, psi′)
end

function TensorKit.lmul!(a::Number, psi::MPSComoving)
    if first(psi.gaugedpos) + 1 == last(psi.gaugedpos)
        #every tensor is either left or right gauged => the bond tensor is centergauged
        @assert !ismissing(psi.bond_tensors[first(psi.gaugedpos)+1])

        lmul!(a, psi.bond_tensors[first(psi.gaugedpos)+1])
    else
        lmul!(a, psi.site_tensors[first(psi.gaugedpos)+1])
    end

    return psi
end

function TensorKit.rmul!(psi::MPSComoving,a::Number)
    if first(psi.gaugedpos) + 1 == last(psi.gaugedpos)
        #every tensor is either left or right gauged => the bond tensor is centergauged
        @assert !ismissing(psi.bond_tensors[first(psi.gaugedpos)+1])

        rmul!(psi.bond_tensors[first(psi.gaugedpos)+1],a)
    else
        rmul!(psi.site_tensors[first(psi.gaugedpos)+1],a)
    end

    return psi
end


function TensorKit.dot(psi1::MPSComoving, psi2::MPSComoving)
    length(psi1) == length(psi2) || throw(ArgumentError("MPS with different length"))
    psi1.left == psi2.left || throw(ArgumentError("left InfiniteMPS is different"))
    psi1.right == psi2.right || throw(ArgumentError("right InfiniteMPS is different"))

    #todo: rewrite this such that it doesn't change gauges
    return tr(_permute_front(psi1.AC[1])' * _permute_front(psi2.AC[1]))
end

function TensorKit.norm(psi::MPSComoving)
    #todo : rewrite this without having to gauge
    if first(psi.gaugedpos) == length(psi)
        #everything is left gauged
        #the bond tensor should be center gauged
        @assert !ismissing(psi.bond_tensors[first(psi.gaugedpos)+1])
        return norm(psi.bond_tensors[first(psi.gaugedpos)+1])
    elseif last(psi.gaugedpos) == 1
        #everything is right gauged
        #the bond tensor should be center gauged
        @assert !ismissing(psi.bond_tensors[1])
        return norm(psi.bond_tensors[1])
    else
        return norm(psi.AC[first(psi.gaugedpos)+1])
    end
end

TensorKit.normalize!(psi::MPSComoving) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::MPSComoving) = normalize!(copy(psi))
