#=
While a finitemps can represent a finitempo using some kind of mapping
they require a different normalization and expectation value ...

You could write code representing it "under the hood" using an A-B structure
but for mpo's, you expect the bond dimension between physical and dual to be really large
which means that the contraction order should be different

Also, 1site tdvp would make no sense (need 2 sites)
and 2site tdvp would correspond with 4 site tdvp (don't have that code)
=#
"
    FiniteMPO(data::Array)

    finite one dimensional mpo
"
mutable struct FiniteMPO{A<:MPOTensor} #<: AbstractMPS
    tensors::Vector{A}
    centerpos::UnitRange{Int} # range of tensors which are not left or right normalized

    function FiniteMPO{A}(tensors::Vector{A},
                            centerpos::UnitRange{Int}) where {A<:MPOTensor}
        space(tensors[1],1) == oneunit(space(tensors[1],1)) ||
            throw(SectorMismatch("Leftmost virtual index of MPS should be trivial."))

        N = length(tensors)
        for n = 1:N-1
            dual(space(tensors[n],3)) == space(tensors[n+1],1) ||
                throw(SectorMismatch("Non-matching virtual index on bond $n."))
        end
        dim(space(tensors[N],3)) == 1 ||
            throw(SectorMismatch("Rightmost virtual index should be one-dimensional."))
        return new{A}(tensors, centerpos)
    end
end

FiniteMPO(tensors::Vector{A},
            centerpos::UnitRange{Int} = 1:length(tensors)) where {A<:MPOTensor} =
    FiniteMPO{A}(tensors, centerpos)

function FiniteMPO(f, P::ProductSpace, args...; kwargs...)
    return FiniteMPO(f, collect(P), args...; kwargs...)
end

function FiniteMPO(f, N::Int, V::VectorSpace, args...; kwargs...)
    return FiniteMPO(f, fill(V, N), args...; kwargs...)
end

function FiniteMPO(f, physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S;
                    left::S = oneunit(maxvirtspace),
                    right::S = oneunit(maxvirtspace)) where {S<:ElementarySpace}
    N = length(physspaces)
    virtspaces = Vector{S}(undef, N+1)
    virtspaces[1] = left
    for k = 2:N
        virtspaces[k] = min(fuse(virtspaces[k-1], fuse(physspaces[k]*physspaces[k]')), maxvirtspace)
    end
    virtspaces[N+1] = right
    for k = N:-1:2
        virtspaces[k] = min(virtspaces[k], fuse(virtspaces[k+1], flip(fuse(physspaces[k]*physspaces[k]'))))
    end
    return FiniteMPO(f, physspaces, virtspaces)
end
function FiniteMPO(f,
                    physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                    virtspaces::Vector{S}) where {S<:ElementarySpace}
    N = length(physspaces)
    length(virtspaces) == N+1 || throw(DimensionMismatch())

    tensors = [TensorMap(f, virtspaces[n] ⊗ physspaces[n], virtspaces[n+1] ⊗ physspaces[n]) for n=1:N]
    if f == randisometry
        return FiniteMPO(tensors, N:N)
    else
        return FiniteMPO(tensors, 1:N)
    end
end

function Base.getproperty(psi::FiniteMPO,prop::Symbol)
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


Base.copy(psi::FiniteMPO) = FiniteMPO(map(copy, psi.tensors), psi.centerpos)
#=
Base.@propagate_inbounds Base.getindex(psi::FiniteMPO, args...) =
    getindex(psi.tensors, args...)
Base.@propagate_inbounds Base.setindex!(psi::FiniteMPO, args...) =
    setindex!(psi.tensors, args...)
=#
Base.length(psi::FiniteMPO) = length(psi.tensors)
Base.size(psi::FiniteMPO, i...) = size(psi.tensors, i...)
#=
Base.firstindex(psi::FiniteMPO, i...) = firstindex(psi.tensors, i...)
Base.lastindex(psi::FiniteMPO, i...) = lastindex(psi.tensors, i...)

Base.iterate(psi::FiniteMPO, i...) = iterate(psi.tensors, i...)
Base.IteratorSize(::Type{<:FiniteMPO}) = Base.HasShape{1}()
Base.IteratorEltype(::Type{<:FiniteMPO}) = Base.HasEltype()
=#
Base.eltype(::Type{FiniteMPO{A}}) where {A<:MPOTensor} = A
Base.similar(psi::FiniteMPO) = FiniteMPO(similar.(psi.tensors))

TensorKit.space(psi::FiniteMPO, n::Integer) = space(psi.A[n], 2)

virtualspace(psi::FiniteMPO, n::Integer) =
    n < length(psi) ? space(psi.A[n+1],1) : dual(space(psi.A[n],3))

r_RR(state::FiniteMPO{T}) where T = isomorphism(Matrix{eltype(T)},space(state.A[end],3)',space(state.A[end],3)')
l_LL(state::FiniteMPO{T}) where T = isomorphism(Matrix{eltype(T)},space(state.A[1],1),space(state.A[1],1))

@bm function expectation_value(ts::FiniteMPO,opp::TensorMap)
    #todo : clean this up
    leftenvs = [Tensor(ones,ComplexF64,space(ts.A[1],1)')];
    rightenvs = [Tensor(ones,ComplexF64,space(ts.A[length(ts)],3)')];

    for i in 1:length(ts)
        @tensor curl[-1] := leftenvs[end][1]*ts.A[i][1,2,-1,2]
        push!(leftenvs,curl);
    end

    for i in length(ts):-1:1
        @tensor curr[-1] := rightenvs[end][1]*ts.A[i][-1,2,1,2]
        push!(rightenvs,curr);
    end

    tor = []
    for i in 1:length(ts)
        cur = @tensor leftenvs[i][1]*ts.A[i][1,2,3,4]*rightenvs[end-i][3]*opp[4,2]
        push!(tor,cur)
    end

    trace = tr(ts);
    tor./trace
end

function max_Ds(f::FiniteMPO)
    Ds = [1 for v in 1:length(f)+1];
    for i in 1:length(f)
        Ds[i+1] = Ds[i]*prod(dim(space(f.A[i],2))*dim(space(f.A[i],4)))
    end

    Ds[end] = 1;
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(dim(space(f.A[i],2))*dim(space(f.A[i],4))))
    end
    Ds
end

"""
    function leftorth!(psi::FiniteMPO, n = length(psi); normalize = true, alg = QRpos())

Bring all MPO tensors of `psi` to the left of site `n` into left orthonormal form, using
the orthogonal factorization algorithm `alg`
"""
function TensorKit.leftorth!(psi::FiniteMPO, n::Integer = length(psi);
                    alg::OrthogonalFactorizationAlgorithm = QRpos(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)
    while first(psi.centerpos) < n
        k = first(psi.centerpos)
        (AL,C) = leftorth!(permute(psi.tensors[k],(1,2,4),(3,));alg=alg)

        psi.tensors[k] = permute(AL,(1,2),(4,3))
        @tensor psi.tensors[k+1][-1 -2;-3 -4] := C[-1,1]*psi.tensors[k+1][1,-2,-3,-4]

        psi.centerpos = k+1:max(k+1,last(psi.centerpos))
    end
    return normalize ? normalize!(psi) : psi
end
function TensorKit.rightorth!(psi::FiniteMPO, n::Integer = 1;
                    alg::OrthogonalFactorizationAlgorithm = LQpos(),
                    normalize::Bool = true)
    @assert 1 <= n <= length(psi)
    while last(psi.centerpos) > n
        k = last(psi.centerpos)
        C, AR = rightorth!(permute(psi.tensors[k],(1,),(2,3,4)); alg = alg)
        psi.tensors[k] = permute(AR,(1,2),(3,4))
        @tensor psi.tensors[k-1][-1 -2;-3 -4] := psi.tensors[k-1][-1,-2,1,-4]*C[1,-3]
        psi.centerpos = min(first(psi.centerpos), k-1):k-1
    end
    return normalize ? normalize!(psi) : psi
end


function Base.:*(psi::FiniteMPO, a::Number)
    psi′ = FiniteMPO(psi.tensors .* one(a) , psi.centerpos)
    return rmul!(psi′, a)
end

function Base.:*(a::Number, psi::FiniteMPO)
    psi′ = FiniteMPO(one(a) .* psi.tensors, psi.centerpos)
    return lmul!(a, psi′)
end

function TensorKit.lmul!(a::Number, psi::FiniteMPO)
    lmul!(a, psi.tensors[first(psi.centerpos)])
    return psi
end

function TensorKit.rmul!(psi::FiniteMPO, a::Number)
    rmul!(psi.tensors[first(psi.centerpos)], a)
    return psi
end

function TensorKit.dot(psi1::FiniteMPO, psi2::FiniteMPO)
    length(psi1) == length(psi2) || throw(ArgumentError("MPO with different length"))

    @tensor ρL[-1;-2] := psi2.A[1][1,2,-2,3]*conj(psi1.A[1][1,2,-1,3])
    for k in 2:length(psi1)
        ρL = transfer_left(ρL, psi2.A[k], psi1.A[k])
    end
    return tr(ρL)
end

"""
    Not normalized according to trace(rho) = 1
    this would've been incompatible with the dot product
"""
function TensorKit.norm(psi::FiniteMPO)

    k = first(psi.centerpos)
    if k == last(psi.centerpos)
        return norm(psi.A[k])
    else
        _, C = leftorth!(permute(psi.A[k],(1,2,4),(3,)))
        k += 1
        while k <= last(psi.centerpos)
            _, C = leftorth!(permute(C * _permute_tail(psi.A[k]),(1,2,4),(3,)))
            k += 1
        end
        return norm(C)
    end
end

TensorKit.normalize!(psi::FiniteMPO) = rmul!(psi, 1/norm(psi))
TensorKit.normalize(psi::FiniteMPO) = normalize!(copy(psi))

function LinearAlgebra.tr(psi::FiniteMPO)
    @tensor v[-1;-2] := psi.A[1][-1,1,-2,1]
    for k in 2:length(psi)
        @tensor v[-1;-2] := v[-1,1]*psi.A[k][1,2,-2,2]
    end
    return tr(v)
end
