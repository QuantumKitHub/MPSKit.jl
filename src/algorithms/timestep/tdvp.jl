"onesite tdvp"
@with_kw struct Tdvp <: Algorithm
    tol::Float64 = Defaults.tol
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
end

"""
    (newstate,newpars) = timestep(state,hamiltonian,dt,alg,pars = params(state,hamiltonian))

    evolves state forward by dt using algorithm alg
"""
@bm function timestep(state::InfiniteMPS, H::Hamiltonian, timestep::Number,alg::Tdvp,parameters::Cache=params(state,H))

    newAs=similar(state.AL)

    for loc in 1:length(state)
        (newAcenter,convhist) = let st=state,pr=parameters
            (newAcenter,convhist) = exponentiate(x->ac_prime(x,loc,st,pr) ,-1im*timestep,st.AC[loc],Lanczos(tol=alg.tol))
        end
        convhist.converged==0 && @info "time evolving ac($loc) failed $(convhist.normres)"

        (newCenter,convhist) = let st=state,pr=parameters
            (newCenter,convhist) = exponentiate(x->c_prime(x,loc, st,pr) , -1im*timestep,st.CR[loc],Lanczos(tol=alg.tol))
        end
        convhist.converged==0 && @info "time evolving c($loc) failed $(convhist.normres)"

        #find Al that best fits these new Acenter and centers
        QAc,_ = leftorth!(newAcenter,alg=TensorKit.Polar())
        Qc,_ = leftorth!(newCenter,alg=TensorKit.Polar())
        @tensor Aleft[-1 -2;-3]:=QAc[-1,-2,1]*conj(Qc[-3,1])

        newAs[loc]     = Aleft
    end

    return InfiniteMPS(newAs; tol = alg.tolgauge, maxiter = alg.maxiter,leftgauged = true,cguess=state.CR[end]),parameters
end

@bm function timestep(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian, timestep::Number,alg::Tdvp,pars=params(state,H))
    #left to right
    for i in 1:(length(state)-1)
        (state.AC[i],convhist)=let pars = pars,state = state
            exponentiate(x->ac_prime(x,i,state,pars),-1im*timestep/2,state.AC[i],Lanczos(tol=alg.tolgauge))
        end

        (state.CR[i],convhist) = let pars = pars,state = state
            exponentiate(x->c_prime(x,i,state,pars),1im*timestep/2,state.CR[i],Lanczos(tol=alg.tolgauge))
        end
    end


    (state.AC[end],convhist)=let pars = pars,state = state
        exponentiate(x->ac_prime(x,length(state),state,pars),-1im*timestep/2,state.AC[end],Lanczos(tol=alg.tolgauge))
    end

    #right to left
    for i in length(state):-1:2
        (state.AC[i],convhist)= let pars=pars, state = state
            exponentiate(x->ac_prime(x,i,state,pars),-1im*timestep/2,state.AC[i],Lanczos(tol=alg.tolgauge))
        end

        (state.CR[i-1],convhist) = let pars = pars, state = state
            exponentiate(x->c_prime(x,i-1,state,pars),1im*timestep/2,state.CR[i-1],Lanczos(tol=alg.tolgauge))
        end
    end

    (state.AC[1],convhist) = let pars=pars, state = state
        exponentiate(x->ac_prime(x,1,state,pars),-1im*timestep/2,state.AC[1],Lanczos(tol=alg.tolgauge))
    end

    return state,pars
end

@bm function timestep(state::FiniteMPO, H::ComAct, timestep::Number,alg::Tdvp,pars=params(state,H))
    @assert false
    #left to right
    for i in 1:(length(state)-1)
        (state[i],convhist)=  let pars=pars, state = state
            exponentiate(x->ac_prime(x,i,state,pars),-1im*timestep/2,state[i],Lanczos(tol=alg.tolgauge))
        end

        (newal,newcenter) = leftorth!(state[i],(1,2,4),(3,),alg=TensorKit.QRpos())
        permute!(state[i],newal,(1,2),(4,3));poison!(pars,i);

        (oldcenter,convhist) =  let pars=pars, state = state
            exponentiate(x->c_prime(x,i,state,pars),1im*timestep/2,newcenter,Lanczos(tol=alg.tolgauge))
        end

        @tensor state[i+1][-1 -2;-3 -4]:=oldcenter[-1,1]*state[i+1][1,-2,-3,-4]
    end

    (state[end],convhist)= let pars=pars, state = state
        exponentiate(x->ac_prime(x,length(state),state,pars),-1im*timestep/2,state[end],Lanczos(tol=alg.tolgauge))
    end

    #right to left
    for i in length(state):-1:2
        (state[i],convhist)= let pars=pars, state = state
            exponentiate(x->ac_prime(x,i,state,pars),-1im*timestep/2,state[i],Lanczos(tol=alg.tolgauge))
        end

        (newcenter,ar) = rightorth!(state[i],(1,),(2,3,4),alg=TensorKit.RQpos())

        permute!(state[i],ar,(1,2),(3,4));poison!(pars,i);

        (oldcenter,convhist) =  let pars=pars, state = state
            exponentiate(x->c_prime(x,i-1,state,pars),1im*timestep/2,newcenter,Lanczos(tol=alg.tolgauge))
        end

        @tensor state[i-1][-1 -2;-3 -4]:=state[i-1][-1,-2,1,-4]*oldcenter[1,-3]
    end

    (state[1],convhist)= let pars=pars, state = state
        exponentiate(x->ac_prime(x,1,state,pars),-1im*timestep/2,state[1],Lanczos(tol=alg.tolgauge))
    end
    return state,pars
end

"twosite tdvp (works for finite mps's)"
@with_kw struct Tdvp2 <: Algorithm
    tol::Float64 = Defaults.tol
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    trscheme = truncerr(1e-3)
end

#twosite tdvp for finite mps
@bm function timestep(state::Union{FiniteMPS,MPSComoving}, H::Hamiltonian, timestep::Number,alg::Tdvp2,pars=params(state,H);rightorthed=false)
    #left to right
    for i in 1:(length(state)-1)
        @tensor ac2[-1 -2;-3 -4]:=state.AC[i][-1,-2,1]*state.AR[i+1][1,-3,-4]

        (nac2,convhist) = let state=state,pars=pars
            exponentiate(x->ac2_prime(x,i,state,pars),-1im*timestep/2,ac2,Lanczos())
        end

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme)

        state.AC[i] = (nal,nc)
        state.AR[i+1] = _permute_front(nar)

        if(i!=(length(state)-1))
            (state.AC[i+1],convhist) = let state=state,pars=pars
                exponentiate(x->ac_prime(x,i+1,state,pars),1im*timestep/2,state.AC[i+1],Lanczos())
            end
        end

    end

    #right to left

    for i in length(state):-1:2
        @tensor ac2[-1 -2;-3 -4]:=state.AL[i-1][-1,-2,1]*state.AC[i][1,-3,-4]

        (nac2,convhist) = let state=state,pars=pars
            exponentiate(x->ac2_prime(x,i-1,state,pars),-1im*timestep/2,ac2,Lanczos())
        end

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme)

        state.AL[i-1] = nal;
        state.AC[i] = (nc,_permute_front(nar));

        if(i!=2)
            (state.AC[i-1],convhist) = let state=state,pars=pars
                exponentiate(x->ac_prime(x,i-1,state,pars),1im*timestep/2,state.AC[i-1],Lanczos())
            end
        end
    end

    return state,pars
end

@bm function timestep(state::FiniteMPO, H::ComAct, timestep::Number,alg::Tdvp2,pars=params(state,H))
    @assert false

    #left to right
    for i in 1:(length(state)-1)
        @tensor ac2[-1 -2 -3;-4 -5 -6]:=state[i][-1,-2,1,-6]*state[i+1][1,-3,-4,-5]

        (nac2,convhist) = let state=state,pars=pars
            exponentiate(x->ac2_prime(x,i,state,pars),-1im*timestep/2,ac2,Lanczos())
        end

        (nal,nc,nar) = tsvd(permute(nac2,(1,2,6),(3,4,5)),trunc=alg.trscheme)
        @tensor nac[-1 -2;-3 -4]:=nc[-1,1]*nar[1,-2,-3,-4]

        state[i]=permute(nal,(1,2),(4,3))
        state[i+1]=nac

        if(i!=(length(state)-1))
            (state[i+1],convhist) = let state=state,pars=pars
                exponentiate(x->ac_prime(x,i+1,state,pars),1im*timestep/2,nac,Lanczos())
            end
        end

    end

    #right to left

    for i in length(state):-1:2
        @tensor ac2[-1 -2 -3;-4 -5 -6]:=state[i-1][-1,-2,1,-6]*state[i][1,-3,-4,-5]

        (nac2,convhist) = let state=state,pars=pars
            exponentiate(x->ac2_prime(x,i-1,state,pars),-1im*timestep/2,ac2,Lanczos())
        end

        (nal,nc,nar) = tsvd(permute(nac2,(1,2,6),(3,4,5)),trunc=alg.trscheme)

        @tensor nac[-1 -2;-3 -4]:=nal[-1,-2,-4,1]*nc[1,-3]

        state[i-1]=nac
        state[i]=permute(nar,(1,2),(3,4))

        if(i!=2)
            (state[i-1],convhist) = let state=state,pars=pars
                exponentiate(x->ac_prime(x,i-1,state,pars),1im*timestep/2,nac,Lanczos())
            end
        end
    end

    return state,pars
end
