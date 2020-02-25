#this thing is useful for statmech and peps
#in principle we could let InfiniteMPS subtype from this thing
#but then we'd have to assert that numrows == 1 everywhere where it doesn't make sense ...

"
    2d extension of InfiniteMPS
"
struct MPSMultiline{A<:GenMPSType,B<:MPSVecType}
    AL::Periodic{A,2}
    AR::Periodic{A,2}
    CR::Periodic{B,2}
    AC::Periodic{A,2}
end

Base.size(arr::MPSMultiline) = size(arr.AL)
Base.size(arr::MPSMultiline,i) = size(arr.AL,i)
Base.length(arr::MPSMultiline) = size(arr,1)
Base.eltype(arr::MPSMultiline) = eltype(arr.AL[1])
Base.lastindex(arr::MPSMultiline,i) = lastindex(arr.AL,i);
Base.similar(st::MPSMultiline) = MPSMultiline(similar(st.AL),similar(st.AR),similar(st.CR),similar(st.AC))

function Base.convert(::Type{MPSMultiline},st::InfiniteMPS)
    AL=Periodic(permutedims(st.AL.data));
    AR=Periodic(permutedims(st.AR.data));
    CR=Periodic(permutedims(st.CR.data));
    AC=Periodic(permutedims(st.AC.data));
    MPSMultiline(AL,AR,CR,AC);
end

function Base.convert(::Type{InfiniteMPS},st::MPSMultiline{A,B}) where {A,B}
    @assert size(st,1) == 1 #otherwise - how would we convert?
    AL=Periodic(st.AL.data[:]);
    AR=Periodic(st.AR.data[:]);
    CR=Periodic(st.CR.data[:]);
    AC=Periodic(st.AC.data[:]);
    InfiniteMPS(AL,AR,CR,AC);
end
#allow users to pass in simple arrays
MPSMultiline(A::Array;tol = Defaults.tolgauge,maxiter = Defaults.maxiter,cguess = [TensorMap(rand, eltype(A[1]), domain(A[i,end]) ← space(A[i,1],1)) for i in 1:size(A,1)]) =
    MPSMultiline(Periodic(A),tol=tol,maxiter=maxiter,cguess=Periodic(cguess))

function MPSMultiline(A::Periodic;tol = Defaults.tolgauge,maxiter = Defaults.maxiter,cguess =  Periodic([TensorMap(rand, eltype(A[1]), domain(A[i,end]) ← space(A[i,1],1)) for i in 1:size(A,1)]))

    ACs = similar(A);ALs = similar(A); ARs = similar(A);
    Cs = Periodic{typeof(cguess[1]),2}(size(A,1),size(A,2));

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
