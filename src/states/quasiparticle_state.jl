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

function rand_quasiparticle(left_gs::InfiniteMPS,right_gs=left_gs;excitation_space = oneunit(space(left_gs.AL[1],1)),momentum=0.0)
    #find the left null spaces for the TNS
    VLs = [adjoint(rightnull(adjoint(v))) for v in left_gs.AL]
    Xs = [TensorMap(rand,eltype(left_gs.AL[1]),space(VLs[loc],3)',excitation_space'*space(right_gs.AR[ loc+1],1)) for loc in 1:length(left_gs)]

    InfiniteQP(left_gs,right_gs,VLs,Xs,momentum)
end

const QP = Union{InfiniteQP,FiniteQP};

utilleg(v::QP) = space(v.Xs[1],2)
Base.copy(a::QP) = copyto!(similar(a),a)
function Base.copyto!(a::QP,b::QP)
    for (i,j) in zip(a.Xs,b.Xs)
        copyto!(i,j)
    end
    a
end
function Base.getproperty(v::QP,s::Symbol)
    if s == :trivial
        return v.left_gs === v.right_gs
    else
        return getfield(v,s)
    end
end

function Base.:-(v::QP,w::QP)
    t = similar(v)
    t.Xs[:] = (v.Xs-w.Xs)[:]
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


function Base.convert(::Type{<:FiniteMPS},v::FiniteQP)
    #very slow and clunky, but shouldn't be performance critical anyway

    function simplefuse(temp)
        frontmap = isomorphism(fuse(space(temp,1)*space(temp,2)),space(temp,1)*space(temp,2))
        backmap = isomorphism(space(temp,5)'*space(temp,4)',fuse(space(temp,5)'*space(temp,4)'))

        @tensor tempp[-1 -2;-3] := frontmap[-1,1,2]*temp[1,2,-2,3,4]*backmap[4,3,-3]
    end


    utl = utilleg(v); ou = oneunit(utl); utsp = ou ⊕ ou;
    upper = isometry(utsp,ou); lower = leftnull(upper);
    upper_I = upper*upper'; lower_I = lower*lower'; uplow_I = upper*lower';

    Ls = v.left_gs.AL[1:end];
    Rs = v.right_gs.AR[1:end];

    #step 0 : fuse the utility leg of B with the first leg of B
    Bs = map(1:length(v)) do i
        t = v[i]
        frontmap = isomorphism(fuse(utl*space(t,1)),utl*space(t,1));
        @tensor tt[-1 -2;-3]:=t[1,-2,2,-3]*frontmap[-1,2,1]
    end

    #step 1 : pass utl through Ls
    passer = isomorphism(utl,utl);
    Ls = map(Ls) do L
        @tensor temp[-1 -2 -3 -4;-5]:=L[-2,-3,-4]*passer[-1,-5]
        simplefuse(temp)
    end

    #step 2 : embed all Ls/Bs/Rs in the same space
    superspaces = map(zip(Ls,Rs)) do (L,R)
        supremum(space(L,1),space(R,1))
    end
    push!(superspaces,supremum(_lastspace(Ls[end])',_lastspace(Rs[end])'))
    for i in 1:(length(v)+1)
        Lf = isometry(superspaces[i],i <= length(v) ? _firstspace(Ls[i]) : _lastspace(Ls[i-1])')
        Rf = isometry(superspaces[i],i <= length(v) ? _firstspace(Rs[i]) : _lastspace(Rs[i-1])')

        if i <= length(v)
            @tensor Ls[i][-1 -2;-3] := Lf[-1,1]*Ls[i][1,-2,-3]
            @tensor Rs[i][-1 -2;-3] := Rf[-1,1]*Rs[i][1,-2,-3]
            @tensor Bs[i][-1 -2;-3] := Lf[-1,1]*Bs[i][1,-2,-3]
        end

        if i>1
            @tensor Ls[i-1][-1 -2;-3] := Ls[i-1][-1 -2;1]*conj(Lf[-3,1])
            @tensor Rs[i-1][-1 -2;-3] := Rs[i-1][-1 -2;1]*conj(Rf[-3,1])
            @tensor Bs[i-1][-1 -2;-3] := Bs[i-1][-1 -2;1]*conj(Rf[-3,1])
        end
    end

    #step 3 : fuse the correct *_I with the correct tensor (and enforce boundary conditions)
    for i in 1:length(v)
        @tensor temp[-1 -2 -3 -4; -5] := Ls[i][-2,-3,-4]*upper_I[-1,-5]
        Ls[i] = simplefuse(temp) * (i<length(v));

        @tensor temp[-1 -2 -3 -4; -5] := Rs[i][-2,-3,-4]*lower_I[-1,-5]
        Rs[i] = simplefuse(temp) * (i>1);

        @tensor temp[-1 -2 -3 -4; -5] := Bs[i][-2,-3,-4]*uplow_I[-1,-5]
        Bs[i] = simplefuse(temp);
    end

    return FiniteMPS(Ls+Rs+Bs)
end
