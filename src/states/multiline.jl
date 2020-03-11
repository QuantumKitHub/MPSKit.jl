#this thing is useful for statmech and peps
#in principle we could let InfiniteMPS subtype from this thing
#but then we'd have to assert that numrows == 1 everywhere where it doesn't make sense ...

"
    2d extension of InfiniteMPS
"
struct MPSMultiline{A<:GenericMPSTensor,B<:MPSBondTensor}
    AL::PeriodicArray{A,2}
    AR::PeriodicArray{A,2}
    CR::PeriodicArray{B,2}
    AC::PeriodicArray{A,2}
end

Base.size(arr::MPSMultiline) = size(arr.AL)
Base.size(arr::MPSMultiline,i) = size(arr.AL,i)
Base.length(arr::MPSMultiline) = size(arr,1)
Base.eltype(arr::MPSMultiline) = eltype(arr.AL[1])
Base.lastindex(arr::MPSMultiline,i) = lastindex(arr.AL,i);
Base.similar(st::MPSMultiline) = MPSMultiline(similar(st.AL),similar(st.AR),similar(st.CR),similar(st.AC))
virtualspace(psi::MPSMultiline, a::Integer,b::Integer) = _firstspace(psi.AL[a,b])
function Base.convert(::Type{MPSMultiline},st::InfiniteMPS)
    AL=PeriodicArray(permutedims(st.AL.data));
    AR=PeriodicArray(permutedims(st.AR.data));
    CR=PeriodicArray(permutedims(st.CR.data));
    AC=PeriodicArray(permutedims(st.AC.data));
    MPSMultiline(AL,AR,CR,AC);
end

function Base.convert(::Type{InfiniteMPS},st::MPSMultiline{A,B}) where {A,B}
    @assert size(st,1) == 1 #otherwise - how would we convert?
    AL=PeriodicArray(st.AL.data[:]);
    AR=PeriodicArray(st.AR.data[:]);
    CR=PeriodicArray(st.CR.data[:]);
    AC=PeriodicArray(st.AC.data[:]);
    InfiniteMPS(AL,AR,CR,AC);
end
#allow users to pass in simple arrays
MPSMultiline(A::Array;tol = Defaults.tolgauge,maxiter = Defaults.maxiter,cguess = [TensorMap(rand, eltype(A[1]), domain(A[i,end]) ← space(A[i,1],1)) for i in 1:size(A,1)]) =
    MPSMultiline(PeriodicArray(A),tol=tol,maxiter=maxiter,cguess=PeriodicArray(cguess))

function MPSMultiline(A::PeriodicArray;tol = Defaults.tolgauge,maxiter = Defaults.maxiter,cguess =  PeriodicArray([TensorMap(rand, eltype(A[1]), domain(A[i,end]) ← space(A[i,1],1)) for i in 1:size(A,1)]))

    ACs = similar(A);ALs = similar(A); ARs = similar(A);
    Cs = PeriodicArray{typeof(cguess[1]),2}(undef,size(A,1),size(A,2));

    for row in 1:size(A,1)
        tal,_,deltal= uniform_leftorth(A[row,:]; tol = tol, maxiter = maxiter, cguess = cguess[row])
        tar,tc,deltar = uniform_rightorth(tal; tol = tol, maxiter = maxiter, cguess = cguess[row])

        ALs[row,:] = tal[:];
        ARs[row,:] = tar[:];
        Cs[row,:] = tc[:];

        for loc = 1:length(tal)
            ACs[row,loc] = ALs[row,loc]*Cs[row,loc]
        end
    end

    return MPSMultiline(ALs,ARs,Cs,ACs)
end

l_RR(state::MPSMultiline,row,loc::Int=1) = @tensor toret[-1;-2]:=state.CR[row,loc-1][1,-2]*conj(state.CR[row,loc-1][1,-1])
l_RL(state::MPSMultiline,row,loc::Int=1) = state.CR[row,loc-1]
l_LR(state::MPSMultiline,row,loc::Int=1) = state.CR[row,loc-1]'
l_LL(state::MPSMultiline{A},row,loc::Int=1) where A= isomorphism(Matrix{eltype(A)}, space(state.AL[row,loc],1),space(state.AL[row,loc],1))

r_RR(state::MPSMultiline{A},row,loc::Int=length(state)) where A= isomorphism(Matrix{eltype(A)},domain(state.AR[row,loc]),domain(state.AR[row,loc]))
r_RL(state::MPSMultiline,row,loc::Int=length(state)) = state.CR[row,loc]'
r_LR(state::MPSMultiline,row,loc::Int=length(state)) = state.CR[row,loc]
r_LL(state::MPSMultiline,row,loc::Int=length(state))= @tensor toret[-1;-2]:=state.CR[row,loc][-1,1]*conj(state.CR[row,loc][-2,1])
