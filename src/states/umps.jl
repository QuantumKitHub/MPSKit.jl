"
    st = MpsCenterGauged(a::Array)

    Type definition of a uniform center gauged mps.
    st.AL[i]*st.CR[i] == st.CR[i-1]*st.AR[i] == st.AC[i]
    st.AL[i] is left unitary
    st.AR[i] is right unitary
"
struct MpsCenterGauged{A<:GenMpsType,B<:MpsVecType}
    AL::Periodic{A,1}
    AR::Periodic{A,1}
    CR::Periodic{B,1}
    AC::Periodic{A,1}
end

Base.size(arr::MpsCenterGauged,i) = size(arr.AL,i)
Base.length(arr::MpsCenterGauged) = size(arr,1)
Base.eltype(arr::MpsCenterGauged) = eltype(arr.AL[1])
Base.copy(m::MpsCenterGauged) = MpsCenterGauged(copy(m.AL),copy(m.AR),copy(m.CR),copy(m.AC));
#Base.circshift(st::MpsCenterGauged,shift::Int) = MpsCenterGauged(circshift(st.AL,shift),circshift(st.AR,shift),circshift(st.CR,shift),circshift(st.AC,shift))
Base.repeat(m::MpsCenterGauged,i::Int) = MpsCenterGauged(repeat(m.AL,i),repeat(m.AR,i),repeat(m.CR,i),repeat(m.AC,i));
Base.similar(st::MpsCenterGauged) = MpsCenterGauged(similar(st.AL),similar(st.AR),similar(st.CR),similar(st.AC))

function MpsCenterGauged(pspaces::M,Dspaces::N;tol::Float64 = Defaults.tolgauge, maxiter::Int64 = Defaults.maxiter,eltype=Defaults.eltype) where M<: AbstractArray{S,1} where N<: AbstractArray{S,1} where S
    MpsCenterGauged([TensorMap(rand,eltype,Dspaces[mod1(i-1,length(Dspaces))]*pspaces[i],Dspaces[i]) for i in 1:length(pspaces)],tol=tol,maxiter=maxiter)
end

#allow users to pass in simple arrays
MpsCenterGauged(A::Array{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int64 = Defaults.maxiter, cguess = TensorMap(rand, eltype(A[1]), domain(A[end]) ← space(A[1],1))) where T<:GenMpsType =  MpsCenterGauged(Periodic(A),tol=tol,maxiter=maxiter,cguess=cguess)
function MpsCenterGauged(A::Periodic{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int64 = Defaults.maxiter, cguess = TensorMap(rand, eltype(A[1]), domain(A[end]) ← space(A[1],1))) where T<:GenMpsType
    #perform the left gauge fixing, remember only Al
    ALs, ncguesses, deltal = leftorth(A[1:end]; tol = tol, maxiter = maxiter, cguess = cguess)

    #perform the right gauge fixing from which we obtain the center matrix
    ARs, Cs, deltar  = rightorth(ALs ; tol = tol, maxiter = maxiter, cguess = ncguesses[end])

    ACs=similar(ARs)
    for loc = 1:size(A,1)
        ACs[loc] = ALs[loc]*Cs[loc]
    end

    return MpsCenterGauged(Periodic(ALs),Periodic(ARs),Periodic(Cs),Periodic(ACs))
end

"
    l_RR(state,location)
    Left dominant eigenvector of the AR-AR transfermatrix
"
l_RR(state::MpsCenterGauged,loc::Int=1) = @tensor toret[-1;-2]:=state.CR[loc-1][1,-2]*conj(state.CR[loc-1][1,-1])

"
    l_RL(state,location)
    Left dominant eigenvector of the AR-AL transfermatrix
"
l_RL(state::MpsCenterGauged,loc::Int=1) = state.CR[loc-1]

"
    l_LR(state,location)
    Left dominant eigenvector of the AL-AR transfermatrix
"
l_LR(state::MpsCenterGauged,loc::Int=1) = state.CR[loc-1]'

"
    l_LL(state,location)
    Left dominant eigenvector of the AL-AL transfermatrix
"
l_LL(state::MpsCenterGauged{A},loc::Int=1) where A= isomorphism(Matrix{eltype(A)}, space(state.AL[loc],1),space(state.AL[loc],1))

"
    r_RR(state,location)
    Right dominant eigenvector of the AR-AR transfermatrix
"
r_RR(state::MpsCenterGauged{A},loc::Int=length(state)) where A= isomorphism(Matrix{eltype(A)},domain(state.AR[loc]),domain(state.AR[loc]))

"
    r_RL(state,location)
    Right dominant eigenvector of the AR-AL transfermatrix
"
r_RL(state::MpsCenterGauged,loc::Int=length(state)) = state.CR[loc]'

"
    r_LR(state,location)
    Right dominant eigenvector of the AL-AR transfermatrix
"
r_LR(state::MpsCenterGauged,loc::Int=length(state)) = state.CR[loc]

"
    r_LL(state,location)
    Right dominant eigenvector of the AL-AL transfermatrix
"
r_LL(state::MpsCenterGauged,loc::Int=length(state))= @tensor toret[-1;-2]:=state.CR[loc][-1,1]*conj(state.CR[loc][-2,1])

function expectation_value(st::MpsCenterGauged,opp::TensorMap)
    dat=[]
    for i in 1:length(st)
        val=@tensor st.AC[i][1,2,3]*opp[4,2]*conj(st.AC[i][1,4,3])
        push!(dat,val)
    end

    return dat
end

function LinearAlgebra.dot(a::MpsCenterGauged,b::MpsCenterGauged)
    init = TensorMap(rand,ComplexF64,space(a.AL[1],1),space(b.AL[1],1))
    num = lcm(length(a),length(b))
    (vals,vecs,convhist) = eigsolve(x->transfer_left(x,b.AL[1:num],a.AL[1:num]),init,1,:LM,Arnoldi())
    convhist.converged == 0 && @info "dot mps not converged"
    return vals[1]
end
