function uniform_leftorth(A::PeriodicArray{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int = Defaults.maxiter,
    cguess = PeriodicArray([isomorphism(Matrix{eltype(A[i])},space(A[i],1),space(A[i],1)) for i in 1:length(A)])) where T <: GenericMPSTensor{S,N1} where {S,N1}
    iteration=1;delta = 2*tol; len = length(A)

    (_,cnew) = leftorth(cguess[1], alg=TensorKit.QRpos())
    normalize!(cnew)


    Al = similar(A)
    Cs = copy(cguess);
    Cs[1] = cnew;

    while iteration<maxiter && delta>tol
        if iteration>10 #when qr starts to fail, start using eigs
            alg=Arnoldi(krylovdim = 30, tol = max(delta*delta,tol/10),maxiter=maxiter)
            #Projection of the current guess onto its largest self consistent eigenvector + isolation of the unitary part

            outp = eigsolve(Cs[1], 1, :LM,alg) do x
                    transfer_left(x,A,Al)
                end

            Cs[1] = leftorth!(outp[2][1],alg=TensorKit.QRpos())[2]
        end

        cold = Cs[1]

        for loc in 1:len
            Al[loc] = permute(Cs[loc]*permute(A[loc],(1,),ntuple(x->x+1,Val{N1}())),ntuple(x->x,Val{N1}()),(N1+1,))
            Al[loc], Cs[loc+1] = leftorth!(Al[loc], alg = QRpos())
            Cs[loc+1]/=norm(Cs[loc+1])
        end

        #update delta
        if domain(cold) == domain(Cs[1]) && codomain(cold) == codomain(Cs[1])
            delta = norm(cold-Cs[1])
        end

        iteration += 1
    end

    delta>tol && @info "leftorth failed to converge $(delta)"

    return Al, Cs, delta
end


function uniform_rightorth(A::PeriodicArray{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int = Defaults.maxiter,
    cguess = PeriodicArray([isomorphism(Matrix{eltype(A[i])},space(A[i],1),space(A[i],1)) for i in 1:length(A)])) where T <: GenericMPSTensor{S,N1} where {S,N1}
    iteration=1; delta = 2*tol; len = length(A)

    (cnew,_) = rightorth(cguess[1], alg=TensorKit.LQpos())
    normalize!(cnew)

    Ar = similar(A)
    Cs = copy(cguess);
    Cs[1] = cnew

    while iteration<maxiter && delta>tol
        if iteration>10#when qr starts to fail, start using eigs
            alg=Arnoldi(krylovdim = 30, tol = max(delta*delta,tol/10),maxiter=maxiter)
            #Projection of the current guess onto its largest self consistent eigenvector + isolation of the unitary part

            outp = eigsolve(Cs[1], 1, :LM,alg) do x
                transfer_right(x,A,Ar)
            end
            Cs[1] = rightorth!(outp[2][1],alg=TensorKit.LQpos())[1]
        end

        cold = Cs[1]
        for loc in len:-1:1
            #@tensor Ar[loc][-1;-2 -3]=A[loc][-1,-2, 1]*Cs[loc][1,-3]
            Ar[loc] = A[loc]*Cs[loc+1]

            temp = permute(Ar[loc],(1,),ntuple(x->x+1,Val{N1}()));

            Cs[loc], temp = rightorth!(temp, alg=LQpos())
            Ar[loc] = permute(temp,ntuple(x->x,Val{N1}()),(N1+1,))
            Cs[loc]/=norm(Cs[loc])
        end

        #update counters and delta
        delta = norm(cold-Cs[1])

        #are we done ?
        iteration += 1
  end

  delta>tol && @info "rightorth failed to converge $(delta)"

  return Ar, Cs, delta
end

#=
#https://arxiv.org/abs/1909.06341
function uniform_leftorth(A::PeriodicArray{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int = Defaults.maxiter,
    cguess = PeriodicArray([isomorphism(Matrix{eltype(A[i])},space(A[i],1),space(A[i],1)) for i in 1:length(A)])) where T <: GenericMPSTensor{S,N1} where {S,N1}

    AL = similar(A);
    C = similar(cguess);
    temp_C = similar(cguess);
    for (i,(cprev,a,cnext)) in enumerate(zip(cguess,A,circshift(cguess,-1)))
        (AL[i],temp_C[i+1]) = leftorth!(_permute_front(cprev*_permute_tail(a*inv(cnext))),alg=QRpos())
        C[i+1] = temp_C[i+1]*cnext
    end

    delta = 2*tol; iteration=1;
    while iteration<maxiter && delta>tol
        delta = 0;

        for (i,(a,c)) in enumerate(zip(copy(AL),copy(temp_C)))
            (AL[i],temp_C[i+1]) = leftorth!(_permute_front(c*_permute_tail(a)),alg = QRpos())
            C[i+1] = temp_C[i+1]*C[i+1]
            normalize!(C[i+1])

            delta = max(delta,norm(temp_C[i+1]-one(temp_C[i+1])*norm(temp_C[i+1],Inf)))
        end

        iteration+=1;

    end
    delta>tol && @info "leftorth failed to converge $(delta)"

    return AL, C, delta
end

function uniform_rightorth(A::PeriodicArray{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int = Defaults.maxiter,
    cguess = PeriodicArray([isomorphism(Matrix{eltype(A[i])},space(A[i],1),space(A[i],1)) for i in 1:length(A)])) where T <: GenericMPSTensor{S,N1} where {S,N1}

    AR = PeriodicArray(_permute_tail.(A));
    C = similar(cguess);
    temp_C = similar(cguess);
    for (i,(cprev,a,cnext)) in reverse(collect(enumerate(zip(cguess,A,circshift(cguess,-1)))))
        (temp_C[i],AR[i]) = rightorth!(inv(cprev)*_permute_tail(a*cnext),alg=LQpos())
        C[i] = cprev*temp_C[i];
    end

    delta = 2*tol; iteration=1;
    while iteration<maxiter && delta>tol
        delta = 0;

        for (i,(a,c)) in enumerate(zip(copy(AR),circshift(temp_C,-1)))
            (temp_C[i],AR[i]) = rightorth!(_permute_tail(_permute_front(a)*c),alg = LQpos())
            C[i] = C[i]*temp_C[i]
            normalize!(C[i])

            delta = max(delta,norm(temp_C[i]-one(temp_C[i])*norm(temp_C[i],Inf))
        end

        iteration+=1;
    end
    delta>tol && @info "rightorth failed to converge $(delta)"

    return PeriodicArray(_permute_front.(AR)), C, delta
end
=#
