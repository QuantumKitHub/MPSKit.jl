#this thing is useful for statmech and peps
#in principle we could let MpsCenterGauged subtype from this thing
#but then we'd have to assert that numrows == 1 everywhere where it doesn't make sense ...

"
    2d extension of MpsCenterGauged
"
struct MpsMultiline{A<:GenMpsType,B<:MpsVecType}
    AL::Periodic{A,2}
    AR::Periodic{A,2}
    CR::Periodic{B,2}
    AC::Periodic{A,2}
end

Base.size(arr::MpsMultiline) = size(arr.AL)
Base.size(arr::MpsMultiline,i) = size(arr.AL,i)
Base.length(arr::MpsMultiline) = size(arr,1)
Base.eltype(arr::MpsMultiline) = eltype(arr.AL[1])
Base.lastindex(arr::MpsMultiline,i) = lastindex(arr.AL,i);

function Base.convert(::Type{MpsMultiline},st::MpsCenterGauged)
    AL=Periodic(permutedims(st.AL.data));
    AR=Periodic(permutedims(st.AR.data));
    CR=Periodic(permutedims(st.CR.data));
    AC=Periodic(permutedims(st.AC.data));
    MpsMultiline(AL,AR,CR,AC);
end

function Base.convert(::Type{MpsCenterGauged},st::MpsMultiline{A,B}) where {A,B}
    @assert size(st,1) == 1 #otherwise - how would we convert?
    AL=Periodic(st.AL.data[:]);
    AR=Periodic(st.AR.data[:]);
    CR=Periodic(st.CR.data[:]);
    AC=Periodic(st.AC.data[:]);
    MpsCenterGauged(AL,AR,CR,AC);
end
#allow users to pass in simple arrays
MpsMultiline(A::Array;tol = Defaults.tolgauge,maxiter = Defaults.maxiter,cguess = [TensorMap(I, eltype(A[1]), domain(A[i,end]) ← space(A[i,1],1)) for i in 1:size(A,1)]) =
    MpsMultiline(Periodic(A),tol=tol,maxiter=maxiter,cguess=Periodic(cguess))

function MpsMultiline(A::Periodic;tol = Defaults.tolgauge,maxiter = Defaults.maxiter,cguess =  Periodic([TensorMap(I, eltype(A[1]), domain(A[i,end]) ← space(A[i,1],1)) for i in 1:size(A,1)]))

    ACs = similar(A);ALs = similar(A); ARs = similar(A);
    Cs = Periodic{typeof(cguess[1]),2}(size(A,1),size(A,2));

    for row in 1:size(A,1)
        tal,_,deltal= leftorth(A[row,:]; tol = tol, maxiter = maxiter, cguess = cguess[row])
        tar,tc,deltar = rightorth(tal; tol = tol, maxiter = maxiter, cguess = cguess[row])

        ALs[row,:] = tal[:];
        ARs[row,:] = tar[:];
        Cs[row,:] = tc[:];

        for loc = 1:length(tal)
            ACs[row,loc] = ALs[row,loc]*Cs[row,loc]
        end
    end

    return MpsMultiline(ALs,ARs,Cs,ACs)
end

l_RR(state::MpsMultiline,row,loc::Int=1) = @tensor toret[-1;-2]:=state.CR[row,loc-1][1,-2]*conj(state.CR[row,loc-1][1,-1])
l_RL(state::MpsMultiline,row,loc::Int=1) = state.CR[row,loc-1]
l_LR(state::MpsMultiline,row,loc::Int=1) = state.CR[row,loc-1]'
l_LL(state::MpsMultiline{A},row,loc::Int=1) where A= TensorMap(I,eltype(A), space(state.AL[row,loc],1),space(state.AL[row,loc],1))

r_RR(state::MpsMultiline{A},row,loc::Int=length(state)) where A= TensorMap(I,eltype(A),domain(state.AR[row,loc]),domain(state.AR[row,loc]))
r_RL(state::MpsMultiline,row,loc::Int=length(state)) = state.CR[row,loc]'
r_LR(state::MpsMultiline,row,loc::Int=length(state)) = state.CR[row,loc]
r_LL(state::MpsMultiline,row,loc::Int=length(state))= @tensor toret[-1;-2]:=state.CR[row,loc][-1,1]*conj(state.CR[row,loc][-2,1])
