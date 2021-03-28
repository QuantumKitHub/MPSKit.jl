"""
onesite infinite dmrg
"""
@with_kw struct Idmrg1{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
end


function find_groundstate(ost::InfiniteMPS, ham::Hamiltonian,alg::Idmrg1,oenvs=environments(ost,ham))
    st = copy(ost);
    envs = IDMRGEnv(ost,oenvs);

    delta::Float64 = 2*alg.tol_galerkin;

    for topit in 1:alg.maxiter
        delta = 0.0;

        curc = st.CR[0];

        for pos = 1:length(st)
            (eigvals,vecs) = @closure eigsolve(st.AC[pos],1,:SR,Arnoldi()) do x
                ac_prime(x,pos,st,envs)
            end

            st.AC[pos] = vecs[1]
            (st.AL[pos],st.CR[pos]) = leftorth(vecs[1]);

            setleftenv!(envs,pos+1,transfer_left(leftenv(envs,pos),ham[pos],st.AL[pos],st.AL[pos]));
        end

        for pos in 1:length(st)-1
            setleftenv!(envs,pos+1,transfer_left(leftenv(envs,pos),ham[pos],st.AL[pos],st.AL[pos]));
        end

        for pos = length(st):-1:1

            (eigvals,vecs) = @closure eigsolve(st.AC[pos],1,:SR,Arnoldi()) do x
                ac_prime(x,pos,st,envs)
            end
            st.AC[pos] = vecs[1]
            (st.CR[pos-1],temp) = rightorth(_permute_tail(vecs[1]));
            st.AR[pos] = _permute_front(temp);

            setrightenv!(envs,pos-1,transfer_right(rightenv(envs,pos),ham[pos],st.AR[pos],st.AR[pos]));
        end

        for pos = length(st):-1:2
            setrightenv!(envs,pos-1,transfer_right(rightenv(envs,pos),ham[pos],st.AR[pos],st.AR[pos]));
        end

        delta = norm(curc-st.CR[0]);
        delta<alg.tol_galerkin && break;
        alg.verbose && @info "idmrg iter $(topit) err $(delta)"
    end

    nst = InfiniteMPS(st.AR[1:end],tol=alg.tol_gauge);
    nenvs = environments(nst,ham,tol=oenvs.tol,maxiter=oenvs.maxiter)
    return nst,nenvs,delta;
end

"""
twosite infinite dmrg
"""
@with_kw struct Idmrg2{} <: Algorithm
    tol_galerkin::Float64 = Defaults.tol
    tol_gauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
    trscheme = truncerr(1e-6);
end

function find_groundstate(ost::InfiniteMPS, ham::Hamiltonian,alg::Idmrg2,oenvs=environments(ost,ham))
    length(ost) < 2 && throw(ArgumentError("unit cell should be >= 2"))

    st = copy(ost);
    envs = IDMRGEnv(ost,oenvs);

    delta::Float64 = 2*alg.tol_galerkin;

    for topit in 1:alg.maxiter
        delta = 0.0;

        curc = st.CR[0];

        #sweep from left to right
        for pos = 1:length(st)-1
            ac2 = st.AC[pos]*_permute_tail(st.AR[pos+1]);

            (eigvals,vecs) = @closure eigsolve(ac2,1,:SR,Arnoldi()) do x
                ac2_prime(x,pos,st,envs)
            end

            (al,c,ar,ϵ) = tsvd(vecs[1],trunc=alg.trscheme,alg=TensorKit.SVD())
            normalize!(c);

            st.AL[pos] = al
            st.CR[pos] = complex(c);
            st.AR[pos+1] = _permute_front(ar);
            st.AC[pos+1] = _permute_front(c*ar);

            setleftenv!(envs,pos+1,transfer_left(leftenv(envs,pos),ham[pos],st.AL[pos],st.AL[pos]));
            setrightenv!(envs,pos,transfer_right(rightenv(envs,pos+1),ham[pos+1],st.AR[pos+1],st.AR[pos+1]))
        end

        #update the edge
        @tensor ac2[-1 -2;-3 -4] := st.AC[end][-1,-2,1]*inv(st.CR[0])[1,2]*st.AL[1][2,-3,3]*st.CR[1][3,-4]
        (eigvals,vecs) = @closure eigsolve(ac2,1,:SR,Arnoldi()) do x
            ac2_prime(x,0,st,envs)
        end
        (al,c,ar,ϵ) = tsvd(vecs[1],trunc=alg.trscheme,alg=TensorKit.SVD())
        normalize!(c);


        st.AC[end] = al*c;
        st.AL[end] = al;
        st.CR[end] = complex(c);
        st.AR[1] = _permute_front(ar);
        st.AC[1] = _permute_front(c*ar);
        st.AL[1] = leftorth(_permute_front(c*ar)*inv(st.CR[1]))[1];

        curc = complex(c);

        #update environments
        setleftenv!(envs,1,transfer_left(leftenv(envs,0),ham[0],st.AL[0],st.AL[0]));
        setrightenv!(envs,0,transfer_right(rightenv(envs,1),ham[1],st.AR[1],st.AR[1]));


        #sweep from right to left
        for pos = length(st)-1:-1:1
            ac2 = st.AL[pos]*_permute_tail(st.AC[pos+1]);

            (eigvals,vecs) = @closure eigsolve(ac2,1,:SR,Arnoldi()) do x
                ac2_prime(x,pos,st,envs)
            end

            (al,c,ar,ϵ) = tsvd(vecs[1],trunc=alg.trscheme,alg=TensorKit.SVD())
            normalize!(c);

            st.AL[pos] = al
            st.AC[pos] = al*c
            st.CR[pos] = complex(c);
            st.AR[pos+1] = _permute_front(ar);
            st.AC[pos+1] = _permute_front(c*ar);

            setrightenv!(envs,pos,transfer_right(rightenv(envs,pos+1),ham[pos+1],st.AR[pos+1],st.AR[pos+1]))
            setleftenv!(envs,pos+1,transfer_left(leftenv(envs,pos),ham[pos],st.AL[pos],st.AL[pos]));
        end

        #update the edge
        @tensor ac2[-1 -2;-3 -4] :=  st.CR[end-1][-1,1]*st.AR[end][1,-2,2]*inv(st.CR[end])[2,3]*st.AC[1][3,-3,-4]
        (eigvals,vecs) = @closure eigsolve(ac2,1,:SR,Arnoldi()) do x
            ac2_prime(x,0,st,envs)
        end
        (al,c,ar,ϵ) = tsvd(vecs[1],trunc=alg.trscheme,alg=TensorKit.SVD())
        normalize!(c);

        st.AR[end] = _permute_front(inv(st.CR[end-1])*_permute_tail(al*c))
        st.AL[end] = al;
        st.CR[end] = complex(c);
        st.AR[1] = _permute_front(ar);
        st.AC[1] = _permute_front(c*ar);

        setleftenv!(envs,1,transfer_left(leftenv(envs,0),ham[0],st.AL[0],st.AL[0]));
        setrightenv!(envs,0,transfer_right(rightenv(envs,1),ham[1],st.AR[1],st.AR[1]));

        #update error
        d1 = Diagonal(convert(Array,curc));
        d2 = Diagonal(convert(Array,complex(c)));
        minl = min(length(d1),length(d2));
        delta = norm(d1[1:minl]-d2[1:minl])

        alg.verbose && @info "idmrg iter $(topit) err $(delta)"

        delta<alg.tol_galerkin && break;

    end

    nst = InfiniteMPS(st.AR[1:end],tol=alg.tol_gauge);
    nenvs = environments(nst,ham,tol=oenvs.tol,maxiter=oenvs.maxiter)
    return nst,nenvs,delta;
end
