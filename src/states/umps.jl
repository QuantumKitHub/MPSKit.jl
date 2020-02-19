"
    st = InfiniteMPS(a::Array)

    Type definition of a uniform center gauged mps.
    st.AL[i]*st.CR[i] == st.CR[i-1]*st.AR[i] == st.AC[i]
    st.AL[i] is left unitary
    st.AR[i] is right unitary
"
struct InfiniteMPS{A<:GenMPSType,B<:MPSVecType}
    AL::PeriodicArray{A,1}
    AR::PeriodicArray{A,1}
    CR::PeriodicArray{B,1}
    AC::PeriodicArray{A,1}
end

Base.size(arr::InfiniteMPS,i) = size(arr.AL,i)
Base.length(arr::InfiniteMPS) = size(arr,1)
Base.eltype(arr::InfiniteMPS) = eltype(arr.AL[1])
Base.copy(m::InfiniteMPS) = InfiniteMPS(copy(m.AL),copy(m.AR),copy(m.CR),copy(m.AC));
#Base.circshift(st::InfiniteMPS,shift::Int) = InfiniteMPS(circshift(st.AL,shift),circshift(st.AR,shift),circshift(st.CR,shift),circshift(st.AC,shift))
Base.repeat(m::InfiniteMPS,i::Int) = InfiniteMPS(repeat(m.AL,i),repeat(m.AR,i),repeat(m.CR,i),repeat(m.AC,i));
Base.similar(st::InfiniteMPS) = InfiniteMPS(similar(st.AL),similar(st.AR),similar(st.CR),similar(st.AC))

function InfiniteMPS(pspaces::AbstractArray{S,1},Dspaces::AbstractArray{S,1};eltype=Defaults.eltype,kwargs...) where S
    InfiniteMPS([TensorMap(rand,eltype,Dspaces[mod1(i-1,length(Dspaces))]*pspaces[i],Dspaces[i]) for i in 1:length(pspaces)];kwargs...)
end

#allow users to pass in simple arrays
function InfiniteMPS(A::AbstractArray{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int64 = Defaults.maxiter, cguess = TensorMap(rand, eltype(A[1]), domain(A[end]) ← space(A[1],1)),leftgauged = false) where T<:GenMPSType
    #perform the left gauge fixing, remember only Al
    if leftgauged
        ALs = A[1:end];
        deltal = 0;
        ncguess = cguess;
    else
        ALs, ncguesses, deltal = uniform_leftorth(A[1:end]; tol = tol, maxiter = maxiter, cguess = cguess)
        ncguess = ncguesses[end];
    end

    #perform the right gauge fixing from which we obtain the center matrix
    ARs, Cs, deltar  = uniform_rightorth(ALs ; tol = tol, maxiter = maxiter, cguess = ncguess)

    ACs=similar(ARs)
    for loc = 1:size(A,1)
        ACs[loc] = ALs[loc]*Cs[loc]
    end

    return InfiniteMPS(PeriodicArray(ALs),PeriodicArray(ARs),PeriodicArray(Cs),PeriodicArray(ACs))
end

"
    l_RR(state,location)
    Left dominant eigenvector of the AR-AR transfermatrix
"
l_RR(state::InfiniteMPS,loc::Int=1) = @tensor toret[-1;-2]:=state.CR[loc-1][1,-2]*conj(state.CR[loc-1][1,-1])

"
    l_RL(state,location)
    Left dominant eigenvector of the AR-AL transfermatrix
"
l_RL(state::InfiniteMPS,loc::Int=1) = state.CR[loc-1]

"
    l_LR(state,location)
    Left dominant eigenvector of the AL-AR transfermatrix
"
l_LR(state::InfiniteMPS,loc::Int=1) = state.CR[loc-1]'

"
    l_LL(state,location)
    Left dominant eigenvector of the AL-AL transfermatrix
"
l_LL(state::InfiniteMPS{A},loc::Int=1) where A= isomorphism(Matrix{eltype(A)}, space(state.AL[loc],1),space(state.AL[loc],1))

"
    r_RR(state,location)
    Right dominant eigenvector of the AR-AR transfermatrix
"
r_RR(state::InfiniteMPS{A},loc::Int=length(state)) where A= isomorphism(Matrix{eltype(A)},domain(state.AR[loc]),domain(state.AR[loc]))

"
    r_RL(state,location)
    Right dominant eigenvector of the AR-AL transfermatrix
"
r_RL(state::InfiniteMPS,loc::Int=length(state)) = state.CR[loc]'

"
    r_LR(state,location)
    Right dominant eigenvector of the AL-AR transfermatrix
"
r_LR(state::InfiniteMPS,loc::Int=length(state)) = state.CR[loc]

"
    r_LL(state,location)
    Right dominant eigenvector of the AL-AL transfermatrix
"
r_LL(state::InfiniteMPS,loc::Int=length(state))= @tensor toret[-1;-2]:=state.CR[loc][-1,1]*conj(state.CR[loc][-2,1])

@bm function expectation_value(st::InfiniteMPS,opp::MPSVecType)
    dat=[]
    for i in 1:length(st)
        val=@tensor st.AC[i][1,2,3]*opp[4,2]*conj(st.AC[i][1,4,3])
        push!(dat,val)
    end

    return dat
end

function LinearAlgebra.dot(a::InfiniteMPS,b::InfiniteMPS)
    init = TensorMap(rand,ComplexF64,space(a.AL[1],1),space(b.AL[1],1))
    num = lcm(length(a),length(b))
    (vals,vecs,convhist) = eigsolve(x->transfer_left(x,b.AL[1:num],a.AL[1:num]),init,1,:LM,Arnoldi())
    convhist.converged == 0 && @info "dot mps not converged"
    return vals[1]
end
