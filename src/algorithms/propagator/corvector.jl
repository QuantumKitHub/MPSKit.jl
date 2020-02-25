#https://www.youtube.com/watch?v=DFdDNOnGjWc
function squaredenvs(state::Union{MPSComoving,FiniteMPS},ham::MPOHamiltonian,pars=params(state,ham))
    nham=conj(ham,transpo=true)*ham

    #to construct the squared caches we will first initialize params
    #then make all data invalid so it will be recalculated
    #then initialize the right caches at the edge
    ncocache=params(state,nham)

    #make sure the dependencies are incorrect, so data will be recalculated
    for i in 1:length(state)
        poison!(ncocache,i)
    end

    #impose the correct boundary conditions (important for comoving mps, should do nothing for finite mps)
    indmap=multmap(conj(ham,transpo=true),ham)

    nleft=leftenv(ncocache,1,state)
    nright=rightenv(ncocache,length(state),state)

    for i in 1:ham.odim
        for j in 1:ham.odim
            @tensor temp[-1 -2 -3;-4]:=leftenv(pars,1,state)[j][1,-3,-4]*conj(leftenv(pars,1,state)[i][1,-2,-1])
            copyto!(nleft[indmap(i,j)].data,temp.data)

            @tensor temp[-1 -2 -3;-4]:=rightenv(pars,length(state),state)[j][-1,-2,1]*conj(rightenv(pars,length(state),state)[i][-4,-3,1])
            copyto!(nright[indmap(i,j)].data,temp.data)
        end
    end

    return nham,ncocache
end

"
https://arxiv.org/pdf/cond-mat/0203500.pdf
"
@bm function dynamicaldmrg(A::Union{MPSComoving,FiniteMPS},z,ham::MPOHamiltonian;init=copy(A),solvtol=Defaults.tol,tol=solvtol*length(A)*2,maxiter=Defaults.maxiter,verbose=Defaults.verbose)
    len=length(A)

    A=rightorth(A);init=rightorth(init)
    w=real(z);eta=imag(z)

    pars1=params(init,ham)
    (ham2,pars2)=squaredenvs(init,ham,pars1)

    #environments for <init | A>
    mixedlenvs=[]
    mixedrenvs=[]
    push!(mixedlenvs,isomorphism(space(A[1],1),space(A[1],1)))
    push!(mixedrenvs,isomorphism(space(A[len],3)',space(A[len],3)'))
    for i in 1:length(A)
        push!(mixedlenvs,transfer_left(mixedlenvs[end],A[i],init[i]))
        push!(mixedrenvs,transfer_right(mixedrenvs[end],A[len-i+1],init[len-i+1]))
    end
    mixedrenvs=reverse(mixedrenvs)

    delta=2*tol

    numit = 0
    while delta>tol && numit<maxiter
        numit+=1
        delta=0

        for i in 1:(length(A)-1)

            #the alternative is using gradient descent, which is at least sure to converge...
            @tensor tos[-1 -2;-3]:=mixedlenvs[i][-1,1]*A[i][1,-2,2]*mixedrenvs[i+1][2,-3]

            (res,convhist)=linsolve(-eta*tos,init[i],GMRES(tol=solvtol)) do x
                y=(eta*eta+w*w)*x
                y-=2*w*ac_prime(x,i,init,pars1)
                y+=ac_prime(x,i,init,pars2)
            end
            delta=max(delta,norm(res-init[i]))
            init[i]=res

            convhist.converged == 0 && @info "r($(i)) failed to converge $(convhist.normres)"

            #move A/init center; update mixed envs
            (A[i],ac) = leftorth(A[i])
            (init[i],ic) = leftorth(init[i])

            @tensor A[i+1][-1 -2;-3]:=ac[-1,1]*A[i+1][1,-2,-3]
            @tensor init[i+1][-1 -2;-3]:=ic[-1,1]*init[i+1][1,-2,-3]

            mixedlenvs[i+1]=transfer_left(mixedlenvs[i],A[i],init[i])
        end

        for i in length(A):-1:2
            @tensor tos[-1 -2;-3]:=mixedlenvs[i][-1,1]*A[i][1,-2,2]*mixedrenvs[i+1][2,-3]

            (res,convhist)=linsolve(-eta*tos,init[i],GMRES(tol=solvtol)) do x
                y=(eta*eta+w*w)*x
                y-=2*w*ac_prime(x,i,init,pars1)
                y+=ac_prime(x,i,init,pars2)
            end

            delta=max(delta,norm(res-init[i]))
            init[i]=res

            convhist.converged == 0 && @info "l($(i)) failed to converge $(convhist.normres)"

            #move A/init center; update mixed envs
            (ac,ta) = rightorth(A[i],(1,),(2,3))
            (ic,ti) = rightorth(init[i],(1,),(2,3))

            A[i] = permute(ta,(1,2),(3,))
            init[i] = permute(ti,(1,2),(3,))

            A[i-1] = A[i-1]*ac
            init[i-1] = init[i-1]*ic


            mixedrenvs[i]=transfer_right(mixedrenvs[i+1],A[i],init[i])
        end

        verbose && println("ddmrg sweep delta : $(delta)")
    end

    a = @tensor mixedlenvs[1][1,2]*A[1][2,3,4]*mixedrenvs[2][4,5]*conj(init[1][1,3,5])
    a = a';
    cb = leftenv(pars1,1,A);
    for i in 1:length(A)
        cb = transfer_left(cb,ham,i,init[i],A[i]);
    end

    b = 0*a
    for i in 1:length(cb)
        b+=@tensor cb[i][1,2,3]*rightenv(pars1,length(A),A)[i][3,2,1];
    end

    v = b/eta-w/eta*a+1im*a
    return v,init
end
