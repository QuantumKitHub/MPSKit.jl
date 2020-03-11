"
    MPSComoving(leftstate,window,rightstate)

    muteable window of tensors on top of an infinite chain
"
mutable struct MPSComoving{Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} <: AbstractMPS
    left_gs::InfiniteMPS{Mtype,Vtype}
    tensors::Array{Mtype,1}
    right_gs::InfiniteMPS{Mtype,Vtype}
    centerpos::UnitRange{Int} # range of tensors which are not left or right normalized

    function MPSComoving(left::InfiniteMPS{Mtype,Vtype},tensors::Array{Mtype,1},right::InfiniteMPS{Mtype,Vtype},centerpos::UnitRange{Int}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor}
        return new{Mtype,Vtype}(left, tensors, right, centerpos)
    end
end

MPSComoving(left::InfiniteMPS{Mtype,Vtype},tensors::Array{Mtype,1},right::InfiniteMPS{Mtype,Vtype}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = MPSComoving(left,tensors,right,1:length(tensors))

Base.copy(state::MPSComoving) = MPSComoving(state.left_gs,map(copy, state.tensors),state.right_gs,state.centerpos)
Base.length(state::MPSComoving)=length(state.tensors)
Base.size(psi::MPSComoving, i...) = size(psi.tensors, i...)
Base.eltype(::Type{MPSComoving{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
Base.similar(psi::MPSComoving) = MPSComoving(psi.left_gs,similar.(psi.tensors),psi.right_gs)

TensorKit.space(psi::MPSComoving{<:MPSTensor}, n::Integer) = space(psi.A[n], 2)
virtualspace(psi::MPSComoving, n::Integer) =
    n < length(psi) ? _firstspace(psi.A[n+1]) : dual(_lastspace(psi.A[n]))


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
    elseif prop == :A
        return AView(psi)
    else
        return getfield(psi,prop)
    end
end


#we need the ability to copy the data from one mpscomoving into another mpscomoving
function Base.copyto!(st1::MPSComoving,st2::MPSComoving)
    for i in 1:length(st1)
        st1[i]=st2[i]
    end
    return st1
end

@bm function expectation_value(state::Union{MPSComoving,FiniteMPS},opp::TensorMap)
    dat=[];

    for i in 1:length(state)
        d = @tensor state.AC[i][1,2,3]*opp[4,2]*conj(state.AC[i][1,4,3])
        push!(dat,d)
    end

    return dat
end

function max_Ds(f::MPSComoving{G}) where G<:GenericMPSTensor{S,N} where {S,N}
    Ds = [dim(space(left_gs.AL[1],1)) for v in 1:length(f)+1];
    for i in 1:length(f)
        Ds[i+1] = Ds[i]*prod(map(x->dim(space(f.A[i],x)),ntuple(x->x+1,Val{N-1}())))
    end

    Ds[end] = dim(space(right_gs.AL[1],1));
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(map(x->dim(space(f.A[i],x)),ntuple(x->x+1,Val{N-1}()))))
    end
    Ds
end

"""
    function leftorth!(psi::MPSComoving, n = length(psi); normalize = true, alg = QRpos())

Bring all MPS tensors of `psi` to the left of site `n` into left orthonormal form, using
the orthogonal factorization algorithm `alg`
"""
function TensorKit.leftorth!(psi::MPSComoving, n::Integer = length(psi);
                    alg::OrthogonalFactorizationAlgorithm = QRpos(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)
    while first(psi.centerpos) < n
        k = first(psi.centerpos)
        AL, C = leftorth!(psi.tensors[k]; alg = alg)
        psi.tensors[k] = AL
        psi.tensors[k+1] = _permute_front(C*_permute_tail(psi.tensors[k+1]))
        psi.centerpos = k+1:max(k+1,last(psi.centerpos))
    end
    return normalize ? normalize!(psi) : psi
end
function TensorKit.rightorth!(psi::MPSComoving, n::Integer = 1;
                    alg::OrthogonalFactorizationAlgorithm = LQpos(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)
    while last(psi.centerpos) > n
        k = last(psi.centerpos)
        C, AR = rightorth!(_permute_tail(psi.tensors[k]); alg = alg)
        psi.tensors[k] = _permute_front(AR)
        psi.tensors[k-1] = psi.tensors[k-1]*C
        psi.centerpos = min(first(psi.centerpos), k-1):k-1
    end
    return normalize ? normalize!(psi) : psi
end


function Base.:*(psi::MPSComoving, a::Number)
    psi′ = MPSComoving(psi.left,psi.tensors .* one(a) ,psi.right, psi.centerpos)
    return rmul!(psi′, a)
end

function Base.:*(a::Number, psi::MPSComoving)
    psi′ = MPSComoving(psi.left,one(a) .* psi.tensors, psi.right,psi.centerpos)
    return lmul!(a, psi′)
end

function TensorKit.lmul!(a::Number, psi::MPSComoving)
    lmul!(a, psi.tensors[first(psi.centerpos)])
    return psi
end

function TensorKit.rmul!(psi::MPSComoving, a::Number)
    rmul!(psi.tensors[first(psi.centerpos)], a)
    return psi
end

#=
    in principle we can take dots between psi1;psi2 when left or right mps differs
=#
function TensorKit.dot(psi1::MPSComoving, psi2::MPSComoving)
    length(psi1) == length(psi2) || throw(ArgumentError("MPS with different length"))
    psi1.left == psi2.left || throw(ArgumentError("left InfiniteMPS is different"))
    psi1.right == psi2.right || throw(ArgumentError("right InfiniteMPS is different"))

    ρL = _permute_front(psi1.A[1])' * _permute_front(psi2.A[1])
    for k in 2:length(psi1)
        ρL = transfer_left(ρL, psi2.A[k], psi1.A[k])
    end
    return tr(ρL)
end

function TensorKit.norm(psi::MPSComoving)
    k = first(psi.centerpos)
    if k == last(psi.centerpos)
        return norm(psi.A[k])
    else
        _, C = leftorth(psi.A[k])
        k += 1
        while k <= last(psi.centerpos)
            _, C = leftorth!(_permute_front(C * _permute_tail(psi.A[k])))
            k += 1
        end
        return norm(C)
    end
end

TensorKit.normalize!(psi::MPSComoving) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::MPSComoving) = normalize!(copy(psi))
