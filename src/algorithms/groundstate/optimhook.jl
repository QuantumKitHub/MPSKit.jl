#use algorithms from optimkit to find the groundstate
function find_groundstate(state::MpsCenterGauged,H::MpoHamiltonian,alg::OptimKit.OptimizationAlgorithm,pars=params(state,H))
    function objfun(x)
        (state,pars) = x;

        #eval = real(sum(expectation_value(state,pars.ham,pars)))


        ac_d = [ac_prime(state.AC[v],v,state,pars) for v in 1:length(state)];

        evals = [dot(state.AC[i],ac_d[i]) for i in 1:length(state)]

        ac_d = [ac_d[i]-evals[i]*state.AC[i] for i in 1:length(state)]

        return real(sum(evals)),ac_d
    end

    function retract(x, cgr, α) # jutho's scheme
        (state,opars) = x;
        ac_d = copy(cgr);

        nacs = [state.AC[i]+α*ac_d[i] for i in 1:length(state)]

        #invert cs
        ics = Periodic([inv(c) for c in state.CR])
        pals = [nacs[i]*ics[i] for i in 1:length(state)]
        pars = [permuteind(ics[i-1]*permuteind(nacs[i],(1,),(2,3)),(1,2),(3,)) for i in 1:length(state)]

        #leftorth rightorth to victory
        (als,cls) = leftorth(pals);
        (ars,crs) = rightorth(pars);

        cs = [cls[i]*state.CR[i]*crs[i] for i in 1:length(state)]
        cs = [c/norm(c) for c in cs]

        #transform gradient
        for i in 1:length(state)
            @tensor nacs[i][-1 -2;-3]:=cls[mod1(i-1,end)][-1,1]*nacs[i][1,-2,2]*crs[i][2,-3]
            lambda=norm(nacs[i])
            nacs[i]/=lambda;

            #consistency check (fail after some time):
            #@show norm(als[i]*cs[i] - nacs[i])#<1e-10
            #@tensor temp[-1 -2;-3]:=cs[mod1(i-1,end)][-1,1]*ars[i][1,-2,-3]
            #@show norm(temp-nacs[i])#<1e-10

            @tensor ac_d[i][-1 -2;-3]:=cls[mod1(i-1,end)][-1,1]*(ac_d[i]/lambda)[1,-2,2]*crs[i][2,-3]
        end

        nstate = MpsCenterGauged(Periodic(als),Periodic(ars),Periodic(cs),Periodic(nacs));

        npars = params(nstate,opars.ham,opars);

        return (nstate,npars),ac_d
    end

    inner(x, v1, v2) = sum([real(dot(x1,x2)) for (x1,x2) in zip(v1,v2)])
    scale!(v, α) = v.*α
    add!(vdst, vsrc, α) = vdst + vsrc.*α

    (x,fx,gx,normgradhistory)=optimize(objfun,(state,pars),alg,retract = retract, inner = inner, scale! = scale!,add! = add!,isometrictransport = false)
    (state,pars) = x;

    return state,pars,normgradhistory

    #(alphas,fs,dfs1,dfs2)=optimtest(objfun, (state,pars), objfun((state,pars))[2]; alpha= 0:0.001:0.1,retract = retract, inner = inner)
end
