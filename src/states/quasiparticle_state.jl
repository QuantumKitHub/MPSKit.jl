#=
Should not be constructed by the user - acts like a vector (used in eigsolve)
I think it makes sense to see these things as an actual state instead of return an array of B tensors (what we used to do)
This will allow us to plot energy density (finite qp) and measure observeables.
=#

struct FiniteQP{S<:FiniteMPS,T1,T2}
    # !(left_gs === right_gs) => domain wall excitation
    left_gs::S
    right_gs::S

    VLs::Vector{T1} # AL' VL = 0 (and VL*X = B)
    Xs::Vector{T2} # contains variational parameters

end

function rand_quasiparticle(left_gs::FiniteMPS,right_gs=left_gs;excitation_space=oneunit(space(left_gs.AL[1]),1))
    #find the left null spaces for the TNS
    VLs = [adjoint(rightnull(adjoint(v))) for v in left_gs.AL]
    Xs = [TensorMap(rand,eltype(left_gs.AL[1]),space(VLs[loc],3)',excitation_space'*space(right_gs.AR[ loc],3)') for loc in 1:length(left_gs)]

    FiniteQP(left_gs,right_gs,VLs,Xs)
end

struct InfiniteQP{S<:InfiniteMPS,T1,T2}
    # !(left_gs === right_gs) => domain wall excitation
    left_gs::S
    right_gs::S

    VLs::Vector{T1} # AL' VL = 0 (and VL*X = B)
    Xs::Vector{T2} # contains variational parameters

    momentum::Float64
end

function rand_quasiparticle(momentum::Float64,left_gs::InfiniteMPS,right_gs=left_gs;excitation_space=oneunit(space(left_gs.AL[1]),1))
    #find the left null spaces for the TNS
    VLs = [adjoint(rightnull(adjoint(v))) for v in left_gs.AL]
    Xs = [TensorMap(rand,eltype(left_gs.AL[1]),space(VLs[loc],3)',excitation_space'*space(right_gs.AR[ loc+1],1)) for loc in 1:length(left_gs)]

    InfiniteQP(left_gs,right_gs,VLs,Xs,momentum)
end

const QP = Union{InfiniteQP,FiniteQP};
function Base.getproperty(v::QP,s::Symbol)
    if s == :trivial
        return v.left_gs === v.right_gs
    else
        return getfield(v,s)
    end
end

LinearAlgebra.dot(v::QP, w::QP) = sum(dot.(v.Xs, w.Xs))
LinearAlgebra.norm(v::QP) = norm(norm.(v.Xs))
Base.length(v::QP) = length(v.Xs)
Base.eltype(v::QP) = eltype(eltype(v.Xs)) # - again debateable, need scaltype
Base.similar(v::InfiniteQP,t=eltype(v)) = InfiniteQP(v.left_gs,v.right_gs,v.VLs,map(e->similar(e,t),v.Xs),v.momentum)
Base.similar(v::FiniteQP,t=eltype(v)) = FiniteQP(v.left_gs,v.right_gs,v.VLs,map(e->similar(e,t),v.Xs))
function LinearAlgebra.mul!(w::QP, a, v::QP)
    @inbounds for (i,j) in zip(w.Xs,v.Xs)
        LinearAlgebra.mul!(i, a, j)
    end
    return w
end

function LinearAlgebra.mul!(w::QP, v::QP, a)
    @inbounds for (i,j) in zip(w.Xs,v.Xs)
        LinearAlgebra.mul!(i, j, a)
    end
    return w
end
function LinearAlgebra.rmul!(v::QP, a)
    for x in v.Xs
        LinearAlgebra.rmul!(x, a)
    end
    return v
end

function LinearAlgebra.axpy!(a, v::QP, w::QP)
    @inbounds for (i,j) in zip(w.Xs,v.Xs)
        LinearAlgebra.axpy!(a, j, i)
    end
    return w
end
function LinearAlgebra.axpby!(a, v::QP, b, w::QP)
    @inbounds for (i,j) in zip(w.Xs,v.Xs)
        LinearAlgebra.axpby!(a, j, b, i)
    end
    return w
end

Base.:*(v::QP, a) = mul!(similar(v),a,v)
Base.:*(a, v::QP) = mul!(similar(v),a,v)

Base.zero(v::QP) = v*0;
Base.getindex(v::QP,i::Int) = v.VLs[i]*v.Xs[i];
function Base.setindex!(v::QP,B,i::Int)
    v.Xs[i] = v.VLs[i]'*B
    v
end
