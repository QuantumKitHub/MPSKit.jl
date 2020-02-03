"""
onesite infinite dmrg
"""
@with_kw struct Idmrg1{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
end


function find_groundstate(st::MpsCenterGauged, ham::Hamiltonian,alg::Idmrg1,pars=params(st,ham))
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
            setleftenv!(pars,i+1,st,mps_apply_transfer_left(leftenv(pars,i,st),ham,i,curu[i]))
        end

        for i in length(st):-1:1

            @tensor curu[i][-1 -2;-3] := curu[i][-1,-2,1]*curc[1,-3]

            (eigvals,vecs) =eigsolve(curu[i],1,:SR,Lanczos()) do x
                ac_prime(x,i,st,pars)
            end

            (curc,temp)=TensorKit.rightorth(vecs[1],(1,),(2,3,))
            curu[i] = permuteind(temp,(1,2),(3,))

            #partially update pars
            setrightenv!(pars,i-1,st,mps_apply_transfer_right(rightenv(pars,i,st),ham,i,curu[i]))
        end

        err = norm(curc-prevc)
        prevc = curc;
        err<alg.tol_galerkin && break;

        alg.verbose && println("idmrg iter $(topit) err $(err)")
    end

    nst = MpsCenterGauged(curu,tol=alg.tol_gauge);
    return nst,params(nst,ham,pars),err;
end

"""
twosite infinite dmrg
"""
@with_kw struct Idmrg2{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    trscheme::TruncationScheme = truncerr(1e-6);
end

function idmrg2_err(oa,ob,a,b)
    v = @tensor oa[1,2,3]*ob[3,4,5]*conj(a[1,2,6])*conj(b[6,4,5])
    return real(1-v*v')
end
#=
#doesn't work either (should update rightenv)
function find_groundstate(st::MpsCenterGauged, ham::Hamiltonian,alg::Idmrg2,pars=params(st,ham))
    curu = Periodic([st.AR[i] for i in 1:length(st)]);

    err = 0.0;

    for topit in 1:alg.maxiter
        err = 0.0

        #sweep from left to right
        for i in 1:(length(st)-1)
            @tensor ac2[-1 -2;-3 -4]:=curu[i][-1,-2,1]*curu[i+1][1,-3,-4]

            (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,i,st,pars)
            end

            (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
            a = U; b = permuteind(S*V,(1,2),(3,));
            err = max(err,idmrg2_err(curu[i],curu[i+1],a,b))
            curu[i] = a; curu[i+1] = b;

            #partially update pars
            setleftenv!(pars,i+1,st,mps_apply_transfer_left(leftenv(pars,i,st),ham,i,curu[i]))
        end

        ALs = copy(curu);
        (ALs[end],CR) = leftorth(ALs[end])

        #sweep from right to left
        for i in (length(st)-1):-1:1
            @tensor ac2[-1 -2;-3 -4]:=curu[i][-1,-2,1]*curu[i+1][1,-3,-4]

            (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,i,st,pars)
            end

            (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
            a = U*S; b = permuteind(V,(1,2),(3,));
            err = max(err,idmrg2_err(curu[i],curu[i+1],a,b))
            curu[i] = a; curu[i+1] = b;

            #partially update pars
            setrightenv!(pars,i,st,mps_apply_transfer_right(rightenv(pars,i+1,st),ham,i+1,curu[i+1]));
        end

        ARs = copy(curu);
        (CL,temp) = rightorth(ARs[1],(1,),(2,3));
        ARs[1] = permuteind(temp,(1,2),(3,));
        ham = circshift(ham,)
        #insert AL[1:end-1] to the left
        #AR[0:end] to the right
        #circshift environments

        alg.verbose && println("idmrg iter $(topit) err $(err)")
        err<alg.tol_galerkin && break;
    end

    nst = MpsCenterGauged(curu),tol=alg.tol_gauge);
    return nst,params(nst,ham,pars),err;
end
=#
#=
#doesn't work either (should update rightenv)
function find_groundstate(st::MpsCenterGauged, ham::Hamiltonian,alg::Idmrg2,pars=params(st,ham))
    curu = Periodic([st.AR[i] for i in 1:length(st)]);
    lamr = Periodic([one(st.CR[i]) for i in 1:length(st)]) #

    err = 0.0;

    for topit in 1:alg.maxiter
        err = 0.0

        #sweep from left to right
        for i in 1:(length(st)-1)
            @tensor ac2[-1 -2;-3 -4]:=curu[i][-1,-2,1]*curu[i+1][1,-3,-4]

            (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,i,st,pars)
            end

            (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
            a = U; b = permuteind(S*V,(1,2),(3,));
            err = max(err,idmrg2_err(curu[i],curu[i+1],a,b))
            lamr[i] = complex(S); curu[i] = a; curu[i+1] = b;

            #partially update pars
            setleftenv!(pars,i+1,st,mps_apply_transfer_left(leftenv(pars,i,st),ham,i,curu[i]))
        end

        #this step happens with incorrect environments; so is incorrect :'(
        #update the last link
        @tensor tar[-1 -2;-3] := lamr[0][-1,1]*curu[1][1,-2,2]*inv(lamr[1])[2,-3]
        @tensor ac2[-1 -2;-3 -4]:=curu[end][-1,-2,1]*tar[1,-3,-4]

        (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
            ac2_prime(x,0,st,pars)
        end

        (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
        a = U*S; b = permuteind(V,(1,2),(3,));
        err = max(err,idmrg2_err(curu[end],tar,a,b))
        curu[end] = U;
        setleftenv!(pars,1,st,mps_apply_transfer_left(leftenv(pars,0,st),ham,0,curu[0]));
        lamr[0] = complex(S); curu[end] = a; @tensor curu[1][-1 -2;-3] := inv(lamr[0])[-1,1]*V[1,-2,2]*lamr[1][2,-3]

        #sweep from right to left
        for i in (length(st)-1):-1:1
            @tensor ac2[-1 -2;-3 -4]:=curu[i][-1,-2,1]*curu[i+1][1,-3,-4]

            (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,i,st,pars)
            end

            (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
            a = U*S; b = permuteind(V,(1,2),(3,));
            err = max(err,idmrg2_err(curu[i],curu[i+1],a,b))
            curu[i] = a; curu[i+1] = b; lamr[i] = complex(S);

            #partially update pars
            setrightenv!(pars,i,st,mps_apply_transfer_right(rightenv(pars,i+1,st),ham,i+1,curu[i+1]));
        end

        #update the last link
        @tensor tal[-1 -2;-3] := inv(lamr[-1])[-1,1]*curu[end][1,-2,2]*lamr[end][2,-3]
        @tensor ac2[-1 -2;-3 -4]:=tal[-1,-2,1]*curu[1][1,-3,-4]

        (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
            ac2_prime(x,0,st,pars)
        end

        (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
        a = U; b = permuteind(S*V,(1,2),(3,));
        err = max(err,idmrg2_err(tal,curu[1],a,b))
        curu[1] = permuteind(V,(1,2),(3,));
        setrightenv!(pars,0,st,mps_apply_transfer_right(rightenv(pars,1,st),ham,1,curu[1]));
        lamr[0] = complex(S); curu[1] = b; @tensor curu[0][-1 -2;-3] := lamr[-1][-1,1]*U[1,-2,2]*inv(lamr[0])[2,-3]

        alg.verbose && println("idmrg iter $(topit) err $(err)")
        err<alg.tol_galerkin && break;
    end

    @tensor curu[1][-1 -2;-3] := lamr[0][-1,1]*curu[1][1,-2,2]*inv(lamr[1])[2,-3]

    nst = MpsCenterGauged(curu,tol=alg.tol_gauge);
    return nst,params(nst,ham,pars),err;
end
=#
#=
#doesn't work
function find_groundstate(st::MpsCenterGauged, ham::Hamiltonian,alg::Idmrg2,pars=params(st,ham))
    L = copy(st.AL);
    R = copy(st.AR);
    C = copy(st.AR);

    err = 0.0;

    for topit in 1:alg.maxiter
        err = 0.0
        #sweep from left to right
        for i in 1:(length(st)-1)
            @tensor ac2[-1 -2;-3 -4]:=C[i][-1,-2,1]*C[i+1][1,-3,-4]

            (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,i,st,pars)
            end

            (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
            a = U; b = permuteind(S*V,(1,2),(3,));
            err = max(err,idmrg2_err(C[i],C[i+1],a,b))
            C[i] = a; C[i+1] = b;

            #partially update pars
            for (a,b) in zip(MPSKit.leftenv(pars,i+1,st),mps_apply_transfer_left(MPSKit.leftenv(pars,i,st),ham,i,C[i]))
                copyto!(a,b)
            end
        end
        #@show err;err=0
        #link update
        @tensor ac2[-1 -2;-3 -4]:=C[end][-1,-2,1]*R[1][1,-3,-4]
        (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
            ac2_prime(x,0,st,pars)
        end
        (U,S,V) = svd(vecs[1],trunc=alg.trscheme)

        a = U; b = permuteind(S*V,(1,2),(3,))
        err = max(err,idmrg2_err(C[end],R[1],a,b))
        C[end] = a;
        for (ta,tb) in zip(MPSKit.leftenv(pars,1,st),mps_apply_transfer_left(MPSKit.leftenv(pars,0,st),ham,0,C[0]))
            copyto!(ta,tb)
        end
        L = C; C = copy(R);
        C[1] = b
        #@show err;err=0
        #sweep from left to right
        for i in 1:(length(st)-1)
            @tensor ac2[-1 -2;-3 -4]:=C[i][-1,-2,1]*C[i+1][1,-3,-4]

            (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,i,st,pars)
            end

            (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
            a = U; b = permuteind(S*V,(1,2),(3,));
            err = max(err,idmrg2_err(C[i],C[i+1],a,b))
            C[i] = a; C[i+1] = b;

            #partially update pars
            for (a,b) in zip(MPSKit.leftenv(pars,i+1,st),mps_apply_transfer_left(MPSKit.leftenv(pars,i,st),ham,i,C[i]))
                copyto!(a,b)
            end
        end
        #@show err;err=0
        #sweep from right to left
        for i in (length(st)-1):-1:1
            @tensor ac2[-1 -2;-3 -4]:=C[i][-1,-2,1]*C[i+1][1,-3,-4]

            (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,i,st,pars)
            end

            (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
            a = U*S; b = permuteind(V,(1,2),(3,));
            err = max(err,idmrg2_err(C[i],C[i+1],a,b))
            C[i] = a; C[i+1] = b;

            #partially update pars
            for (a,b) in zip(MPSKit.rightenv(pars,i,st),mps_apply_transfer_right(MPSKit.rightenv(pars,i+1,st),ham,i,C[i+1]))
                copyto!(a,b)
            end
        end
        #@show err;err=0
        #link update
        @tensor ac2[-1 -2;-3 -4]:=L[end][-1,-2,1]*C[1][1,-3,-4]
        (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
            ac2_prime(x,0,st,pars)
        end
        (U,S,V) = svd(vecs[1],trunc=alg.trscheme)

        a = U*S
        b = permuteind(V,(1,2),(3,));
        err = max(err,idmrg2_err(L[end],C[1],a,b))
        C[1] = b
        for (ta,tb) in zip(MPSKit.rightenv(pars,0,st),mps_apply_transfer_right(rightenv(pars,1,st),ham,1,C[1]))
            copyto!(ta,tb)
        end
        R = C; C = copy(L)
        C[end] = a
        #@show err;err=0
        #sweep from right to left
        for i in (length(st)-1):-1:1
            @tensor ac2[-1 -2;-3 -4]:=C[i][-1,-2,1]*C[i+1][1,-3,-4]

            (eigvals,vecs) =eigsolve(ac2,1,:SR,Lanczos()) do x
                ac2_prime(x,i,st,pars)
            end

            (U,S,V) = svd(vecs[1],trunc=alg.trscheme)
            a = U*S; b = permuteind(V,(1,2),(3,));
            err = max(err,idmrg2_err(C[i],C[i+1],a,b))
            C[i] = a; C[i+1] = b;

            #partially update pars
            for (a,b) in zip(MPSKit.rightenv(pars,i,st),mps_apply_transfer_right(rightenv(pars,i+1,st),ham,i,C[i+1]))
                copyto!(a,b)
            end
        end
        #@show err;
        err<alg.tol_galerkin && break;

        alg.verbose && println("idmrg iter $(topit) err $(err)")
    end

    nst = MpsCenterGauged(R,tol=alg.tol_gauge);
    return nst,params(nst,ham,pars),err;
end
=#
