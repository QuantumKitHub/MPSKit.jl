"""
onesite infinite dmrg
"""
@with_kw struct Idmrg1{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
end


function find_groundstate(st::MpsCenterGauged, ham::Hamiltonian,alg::Idmrg1,opars=params(st,ham))
    pars = SimpleEnv(st,opars);

    curu = [st.AR[i] for i in 1:length(st)];
    prevc = st.CR[0];

    err = 0.0;

    for topit in 1:alg.maxiter
        curc = copy(prevc);

        for i in 1:length(st)
            @tensor curu[i][-1 -2;-3] := curc[-1,1]*curu[i][1,-2,-3]

            (eigvals,vecs) =eigsolve(curu[i],1,:SR,Lanczos()) do x
                ac_prime(x,i,st,pars)
            end

            (curu[i],curc)=TensorKit.leftorth!(vecs[1])

            #partially update pars
            setleftenv!(pars,i+1,st,transfer_left(leftenv(pars,i,st),ham,i,curu[i]))
        end

        for i in length(st):-1:1

            @tensor curu[i][-1 -2;-3] := curu[i][-1,-2,1]*curc[1,-3]

            (eigvals,vecs) =eigsolve(curu[i],1,:SR,Lanczos()) do x
                ac_prime(x,i,st,pars)
            end

            (curc,temp)=TensorKit.rightorth(vecs[1],(1,),(2,3,))
            curu[i] = permute(temp,(1,2),(3,))

            #partially update pars
            setrightenv!(pars,i-1,st,transfer_right(rightenv(pars,i,st),ham,i,curu[i]))
        end

        err = norm(curc-prevc)
        prevc = curc;
        err<alg.tol_galerkin && break;

        alg.verbose && println("idmrg iter $(topit) err $(err)")
    end

    nst = MpsCenterGauged(curu,tol=alg.tol_gauge);
    return nst,params(nst,ham,tol=opars.tol,maxiter=opars.maxiter),err;
end
