"
    MPSComoving(leftstate,window,rightstate)

    muteable window of tensors on top of an infinite chain
"
mutable struct MPSComoving{Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor}
    left_gs::InfiniteMPS{Mtype,Vtype}
    middle::Array{Mtype,1}
    right_gs::InfiniteMPS{Mtype,Vtype}
    centerpos::UnitRange{Int} # range of tensors which are not left or right normalized

    function MPSComoving{Mtype,Vtype}(left::InfiniteMPS{Mtype,Vtype},middle::Array{Mtype,1},right::InfiniteMPS{Mtype,Vtype},centerpos::UnitRange{Int}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor}
        return new{Mtype,Vtype}(left, middle, right, centerpos)
    end
end

MPSComoving(left::InfiniteMPS{Mtype,Vtype},middle::Array{Mtype,1},right::InfiniteMPS{Mtype,Vtype}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = MPSComoving(left,middle,right,1:length(middle))

Base.copy(state::MPSComoving) = MPSComoving(state.left_gs,map(copy, state.middle),state.right_gs,state.centerpos)

#maybe we should allow getindex outside of middle?
Base.@propagate_inbounds Base.getindex(psi::MPSComoving, args...) =
    getindex(psi.middle, args...)
Base.@propagate_inbounds Base.setindex!(psi::MPSComoving, args...) =
    setindex!(psi.middle, args...)

Base.length(state::MPSComoving)=length(state.middle)
Base.size(psi::MPSComoving, i...) = size(psi.middle, i...)
Base.firstindex(psi::MPSComoving, i...) = firstindex(psi.middle, i...)
Base.lastindex(psi::MPSComoving, i...) = lastindex(psi.middle, i...)
Base.iterate(psi::MPSComoving, i...) = iterate(psi.middle, i...)

Base.IteratorSize(::Type{<:MPSComoving}) = Base.HasShape{1}()
Base.IteratorEltype(::Type{<:MPSComoving}) = Base.HasEltype()
Base.eltype(::Type{MPSComoving{Mtype,Vtype}}) where {Mtype<:GenericMPSTensor,Vtype<:MPSBondTensor} = Mtype
Base.similar(psi::MPSComoving) = MPSComoving(psi.left_gs,similar.(psi.tensors),psi.right_gs)

TensorKit.space(psi::MPSComoving{<:MPSTensor}, n::Integer) = space(psi[n], 2)
virtualspace(psi::MPSComoving, n::Integer) =
    n < length(psi) ? _firstspace(psi[n+1]) : dual(_lastspace(psi[n]))


r_RR(state::MPSComoving)=r_RR(state.right_gs,length(state))
l_LL(state::MPSComoving)=l_LL(state.left_gs,1)

#we need the ability to copy the data from one mpscomoving into another mpscomoving
function Base.copyto!(st1::MPSComoving,st2::MPSComoving)
    for i in 1:length(st1)
        st1[i]=st2[i]
    end
    return st1
end

@bm function expectation_value(state::Union{MPSComoving,FiniteMPS},opp::TensorMap;leftorthed=false)
    if(!leftorthed)
        state=leftorth(state)
    end

    dat=[]

    for i in length(state):-1:1
        d=@tensor state[i][1,2,3]*opp[4,2]*conj(state[i][1,4,3])
        push!(dat,d)

        if i!=1
            (c,ar)=TensorKit.rightorth(state[i],(1,),(2,3))
            state[i]=permute(ar,(1,2),(3,))
            state[i-1]=state[i-1]*c
        end
    end

    return reverse(dat)
end

function max_Ds(f::MPSComoving{G}) where G<:GenericMPSTensor{S,N} where {S,N}
    Ds = [dim(space(left_gs.AL[1],1)) for v in 1:length(f)+1];
    for i in 1:length(f)
        Ds[i+1] = Ds[i]*prod(map(x->dim(space(f[i],x)),ntuple(x->x+1,Val{N-1}())))
    end

    Ds[end] = dim(space(right_gs.AL[1],1));
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(map(x->dim(space(f[i],x)),ntuple(x->x+1,Val{N-1}()))))
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
        AL, C = leftorth!(psi[k]; alg = alg)
        psi[k] = AL
        psi[k+1] = _permute_front(C*_permute_tail(psi[k+1]))
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
        C, AR = rightorth!(_permute_tail(psi[k]); alg = alg)
        psi[k] = _permute_front(AR)
        psi[k-1] = psi[k-1]*C
        psi.centerpos = min(first(psi.centerpos), k-1):k-1
    end
    return normalize ? normalize!(psi) : psi
end
