"
https://arxiv.org/pdf/cond-mat/0203500.pdf
"
function dynamicaldmrg(A::AbstractFiniteMPS,z,ham::MPOHamiltonian;init=copy(A),solvtol=Defaults.tol,tol=solvtol*length(A)*2,maxiter=Defaults.maxiter,verbose=Defaults.verbose)
    w=real(z);eta=imag(z)

    envs1 = environments(init,ham) #environments for h
    (ham2,envs2) = squaredenvs(init,ham,envs1) #environments for h^2
    mixedenvs = environments(init,A); #environments for <init | A>

    delta=2*tol

    numit = 0
    while delta>tol && numit<maxiter
        numit+=1
        delta=0

        for i in [1:(length(A)-1);length(A):-1:2]

            @plansor tos[-1 -2;-3] := leftenv(mixedenvs,i,init)[-1;1]*A.AC[i][1 -2;2]*rightenv(mixedenvs,i,init)[2;-3]


            H1_AC = ∂∂AC(i,init,ham,envs1);
            H2_AC = ∂∂AC(i,init,ham2,envs2);
            H_AC = LinearCombination((H1_AC,H2_AC),(-2*w,1));
            (res,convhist) = linsolve(H_AC,-eta*tos,init.AC[i],GMRES(tol=solvtol),(eta*eta+w*w),1);

            delta = max(delta,norm(res-init.AC[i]))
            init.AC[i] = res

            convhist.converged == 0 && @info "($(i)) failed to converge $(convhist.normres)"
        end

        verbose && @info "ddmrg sweep delta : $(delta)"
    end

    a = @plansor leftenv(mixedenvs,1,init)[-1;1]*A.AC[1][1 -2;2]*rightenv(mixedenvs,1,init)[2;-3]*conj(init.AC[1][-1 -2;-3])
    a = a';

    cb = leftenv(envs1,1,A)*TransferMatrix(init.AL,ham[1:length(A.AL)],A.AL);

    b = 0*a
    for i in 1:length(cb)
        b+= @plansor cb[i][1 2;3]*init.CR[end][3;4]*rightenv(envs1,length(A),A)[i][4 2;5]*conj(A.CR[end][1;5]);
    end

    v = b/eta-w/eta*a+1im*a
    return v,init
end

function squaredenvs(state::AbstractFiniteMPS,ham::MPOHamiltonian,envs=environments(state,ham))
    nham = conj(ham)*ham

    #to construct the squared caches we will first initialize environments
    #then make all data invalid so it will be recalculated
    #then initialize the right caches at the edge
    ncocache = environments(state,nham)

    #make sure the dependencies are incorrect, so data will be recalculated
    for i in 1:length(state)
        poison!(ncocache,i)
    end

    #impose the correct boundary conditions (important for comoving mps, should do nothing for finite mps)
    indmap = LinearIndices((ham.odim,ham.odim));

    nleft = leftenv(ncocache,1,state)
    nright = rightenv(ncocache,length(state),state)

    for i in 1:ham.odim,j in 1:ham.odim
        f1 = isomorphism(space(nleft[indmap[i,j]],2),space(leftenv(envs,1,state)[i],2)'*space(leftenv(envs,1,state)[j],2))
        @plansor nleft[indmap[i,j]][-1 -2;-3]:=leftenv(envs,1,state)[j][1 3;-3]*conj(leftenv(envs,1,state)[i][1 2;-1])*f1[-2;2 3]

        f2 = isomorphism(space(nright[indmap[i,j]],2),space(rightenv(envs,length(state),state)[j],2)*space(rightenv(envs,length(state),state)[i],2)');
        @plansor nright[indmap[i,j]][-1 -2;-3]:=rightenv(envs,length(state),state)[j][-1 2;1]*conj(rightenv(envs,length(state),state)[i][-3 3;1])*f2[-2;2 3]
    end

    return nham,ncocache
end
