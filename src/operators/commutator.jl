#implements 'effecient' code for the commutator acting on an mpo
#code is a bit ugly, and can be speed up

"
    ComAct(ham1,ham2)

    Acts on an mpo with mpo hamiltonian 'ham1' from below + 'ham2' from above.
    Can therefore represent the (anti) commutator.
"
struct ComAct{T1<:MPOHamiltonian,T2<:MPOHamiltonian} <: Hamiltonian
    below::T1
    above::T2
end

commutator(H::MPOHamiltonian) = ComAct(H,-1*H)
anticommutator(H::MPOHamiltonian) = ComAct(H,H)

function Base.getproperty(h::ComAct,f::Symbol)
    if f==:odim
        return (h.below.odim+h.above.odim)::Int
    elseif f==:domspaces
        return PeriodicArray([[h.below.domspaces[i];h.above.domspaces[i]] for i in 1:h.period])
    elseif f==:imspaces
        return PeriodicArray([[h.below.imspaces[i];h.above.imspaces[i]] for i in 1:h.period])
    elseif f==:period
        @assert h.above.period == h.below.period
        return h.below.period::Int
    else
        return getfield(h,f)
    end
end

Base.:+(a::ComAct,b::AbstractArray) = ComAct(a.below+b./2,a.above+b./2);
Base.:+(b::AbstractArray,a::ComAct) = a+b;
Base.:+(a::ComAct,b::ComAct) = ComAct(a.below+b.below,a.above+b.above)
Base.:*(a::ComAct,b::Number) = ComAct(a.below*b,a.above*b);
Base.:*(b::Number,a::ComAct) = a*b;
Base.:-(a::ComAct,b::AbstractArray) = -1*(b-a);
Base.:-(a,b::ComAct) = a+-1*b;

isbelow(a::ComAct,ind::Int) = ind<=a.below.odim;
keys(a::ComAct,pos) =   [
                        [ (i[1],i[2]) for i in keys(a.below,pos)];
                        [ (i[1],i[2]).+a.below.odim for i in keys(a.above,pos)]
                        ]


function Base.getindex(a::ComAct,pos::Int,i::Int,j::Int)
    if i<=a.below.odim && j<=a.below.odim
        return a.below[pos,i,j]
    elseif i>a.below.odim && j>a.below.odim
        return a.above[pos,i-a.below.odim,j-a.below.odim]
    else
        @assert false
    end
end

function transfer_left(vec::T,ham::ComAct,pos,A,Ab=A) where T
    toreturn = [TensorMap(zeros,eltype(vec[1]),space(A,4)'*ham.imspaces[pos][i],space(A,4)') for i in 1:ham.odim]::T

    for (i,j) in keys(ham,pos)
        opp = ham[pos,i,j]
        if isbelow(ham,i)
            @tensor toreturn[j][-1 -2;-3] += vec[i][1,3,4]*A[4,5,6,-3]*opp[3,2,-2,5]*conj(Ab[1,2,6,-1])
        else
            @tensor toreturn[j][-1 -2;-3] += vec[i][6,2,1]*A[1,5,3,-3]*opp[2,3,-2,4]*conj(Ab[6,5,4,-1])
        end
    end

    return toreturn
end
function transfer_right(vec::T,ham::ComAct,pos,A,Ab=A) where T
    toreturn = [TensorMap(zeros,eltype(vec[1]),space(A,1)*ham.domspaces[pos][i],space(A,1)) for i in 1:ham.odim]::T

    for (i,j) in keys(ham,pos)
        opp = ham[pos,i,j]
        if isbelow(ham,i)
            @tensor toreturn[i][-1 -2;-3] += vec[j][4,3,1]*A[-1,5,6,4]*opp[-2,2,3,5]*conj(Ab[-3,2,6,1])
        else
            @tensor toreturn[i][-1 -2;-3] += vec[j][1,2,6]*A[-1,5,3,1]*opp[-2,3,2,4]*conj(Ab[-3,5,4,6])
        end
    end

    return toreturn
end
