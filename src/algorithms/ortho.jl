function uniform_leftorth(A::Array{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int = Defaults.maxiter, cguess= TensorMap(rand, eltype(A[1]), domain(A[end]) ← space(A[1],1))) where T <: GenericMPSTensor{S,N1} where {S,N1}
    iteration=1;delta = 2*tol; len = length(A)

    (_,cnew) = leftorth(cguess, alg=TensorKit.QRpos())
    normalize!(cnew)


    Al = similar.(A)
    Cs = fill(cnew,len)

    while iteration<maxiter && delta>tol
        if iteration>10 #when qr starts to fail, start using eigs
            alg=Arnoldi(krylovdim = 30, tol = max(delta*delta,tol/10),maxiter=maxiter)
            #Projection of the current guess onto its largest self consistent eigenvector + isolation of the unitary part

            outp = eigsolve(Cs[end], 1, :LM,alg) do x
                    transfer_left(x,A,Al)
                end

            Cs[end] = leftorth!(outp[2][1],alg=TensorKit.Polar())[2]
        end

        cold = Cs[len]

        for loc in 1:len
            prev= mod1(loc-1,len)

            Al[loc] = permute(Cs[prev]*permute(A[loc],(1,),ntuple(x->x+1,Val{N1}())),ntuple(x->x,Val{N1}()),(N1+1,))
            Al[loc], Cs[loc] = leftorth!(Al[loc], alg=TensorKit.Polar())
            Cs[loc]/=norm(Cs[loc])
        end

        #update delta
        if domain(cold) == domain(Cs[len]) && codomain(cold) == codomain(Cs[len])
            delta = norm(cold-Cs[len],Inf)
        end

        iteration += 1
    end

    delta>tol && @info "leftorth failed to converge $(delta)"

    return Al, Cs, delta
end


function uniform_rightorth(A::Array{T,1}; tol::Float64 = Defaults.tolgauge, maxiter::Int = Defaults.maxiter, cguess = TensorMap(rand, eltype(A[1]), domain(A[end]) ← space(A[1],1))) where T <: GenericMPSTensor{S,N1} where {S,N1}
    iteration=1; delta = 2*tol; len = length(A)

    (cnew,_) = rightorth!(cguess, alg=TensorKit.RQpos())
    normalize!(cnew)

    Ar = similar.(A)
    Cs = fill(cnew,len)

    while iteration<maxiter && delta>tol
        if iteration>10#when qr starts to fail, start using eigs
            alg=Arnoldi(krylovdim = 30, tol = max(delta*delta,tol/10),maxiter=maxiter)
            #Projection of the current guess onto its largest self consistent eigenvector + isolation of the unitary part

            outp = eigsolve(Cs[end], 1, :LM,alg) do x
                transfer_right(x,A,Ar)
            end
            Cs[end] = rightorth!(outp[2][1],alg=TensorKit.Polar())[1]
        end

        cold = Cs[end]
        for loc in len:-1:1
            prev=mod1(loc-1,len)

            #@tensor Ar[loc][-1;-2 -3]=A[loc][-1,-2, 1]*Cs[loc][1,-3]
            Ar[loc] = A[loc]*Cs[loc]

            temp = permute(Ar[loc],(1,),ntuple(x->x+1,Val{N1}()));
            Cs[prev], temp = rightorth!(temp, alg=TensorKit.Polar())
            Ar[loc] = permute(temp,ntuple(x->x,Val{N1}()),(N1+1,))
            Cs[prev]/=norm(Cs[prev])
        end

        #update counters and delta
        delta = norm(cold-Cs[end],Inf)

        #are we done ?
        iteration += 1
  end

  delta>tol && @info "rightorth failed to converge $(delta)"

  return Ar, Cs, delta
end
