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
function timestep(state::MpsCenterGauged, H::Hamiltonian, timestep::Number,alg::Tdvp,parameters::Cache=params(state,H))

    newAs=similar(state.AL)

    for loc in 1:length(state)
        newAcenter = let st=state,pr=parameters
            (newAcenter,convhist) = exponentiate(x->ac_prime(x,loc,st,pr) ,-1im*timestep,st.AC[loc],Lanczos(tol=alg.tol))
            convhist.converged==0 && @info "time evolving ac($loc) failed $(convhist.normres)"
            newAcenter
        end

        newCenter = let st=state,pr=parameters
            (newCenter,convhist) = exponentiate(x->c_prime(x,loc, st,pr) , -1im*timestep,st.CR[loc],Lanczos(tol=alg.tol))
            convhist.converged==0 && @info "time evolving c($loc) failed $(convhist.normres)"
            newCenter
        end

        #find Al that best fits these new Acenter and centers
        QAc,_ = TensorKit.leftorth!(newAcenter,alg=TensorKit.Polar())
        Qc,_ = TensorKit.leftorth!(newCenter,alg=TensorKit.Polar())
        @tensor Aleft[-1 -2;-3]:=QAc[-1,-2,1]*conj(Qc[-3,1])

        newAs[loc]     = Aleft
    end

    return MpsCenterGauged(newAs; tol = alg.tolgauge, maxiter = alg.maxiter),parameters
end

#assumes right orthonormalization, will partly overwrite things in state
function timestep(state::Union{FiniteMps,MpsComoving}, H::Hamiltonian, timestep::Number,alg::Tdvp,pars=params(state,H))
    #left to right
    for i in 1:(length(state)-1)

        (state[i],convhist)=let pars = pars,state = state
            exponentiate(x->ac_prime(x,i,state,pars),-1im*timestep/2,state[i],Lanczos(tol=alg.tolgauge))
        end

        #move to the right
        (state[i],newcenter) = TensorKit.leftorth!(state[i],alg=TensorKit.QRpos());poison!(pars,i);

        (oldcenter,convhist) = let pars = pars,state = state
            exponentiate(x->c_prime(x,i,state,pars),1im*timestep/2,newcenter,Lanczos(tol=alg.tolgauge))
        end

        @tensor state[i+1][-1 -2;-3]:=oldcenter[-1,1]*state[i+1][1,-2,-3]

    end


    (state[end],convhist)=let pars = pars,state = state
        exponentiate(x->ac_prime(x,length(state),state,pars),-1im*timestep/2,state[end],Lanczos(tol=alg.tolgauge))
    end

    #right to left
    for i in length(state):-1:2

        (state[i],convhist)= let pars=pars, state = state
            exponentiate(x->ac_prime(x,i,state,pars),-1im*timestep/2,state[i],Lanczos(tol=alg.tolgauge))
        end

        #in this case we need to split newAcenter in a left gauge fixed part (that will remain at spot i) and a center that will move to the next site
        (newcenter,ar) = TensorKit.rightorth(state[i],(1,),(2,3,),alg=TensorKit.RQpos())
        permute!(state[i],ar,(1,2),(3,));poison!(pars,i);

        (oldcenter,convhist) = let pars = pars, state = state
            exponentiate(x->c_prime(x,i-1,state,pars),1im*timestep/2,newcenter,Lanczos(tol=alg.tolgauge))
        end
        @tensor state[i-1][-1 -2;-3]:=state[i-1][-1,-2,1]*oldcenter[1,-3]
    end

    (state[1],convhist) = let pars=pars, state = state
        exponentiate(x->ac_prime(x,1,state,pars),-1im*timestep/2,state[1],Lanczos(tol=alg.tolgauge))
    end

    return state,pars
end

function timestep(state::FiniteMpo, H::ComAct, timestep::Number,alg::Tdvp,pars=params(state,H))
    #left to right
    for i in 1:(length(state)-1)
        (state[i],convhist)=  let pars=pars, state = state
            exponentiate(x->ac_prime(x,i,state,pars),-1im*timestep/2,state[i],Lanczos(tol=alg.tolgauge))
        end

        (newal,newcenter) = TensorKit.leftorth(state[i],(1,2,4),(3,),alg=TensorKit.QRpos())
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

        (newcenter,ar) = TensorKit.rightorth(state[i],(1,),(2,3,4),alg=TensorKit.RQpos())

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
function timestep(state::Union{FiniteMps,MpsComoving}, H::Hamiltonian, timestep::Number,alg::Tdvp2,pars=params(state,H);rightorthed=false)
    if(!rightorthed)
        state = rightorth!(state)
    end

    #left to right
    for i in 1:(length(state)-1)
        @tensor ac2[-1 -2;-3 -4]:=state[i][-1,-2,1]*state[i+1][1,-3,-4]

        (nac2,convhist) = exponentiate(x->ac2_prime(x,i,state,pars),-1im*timestep/2,ac2,Lanczos())

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme)
        @tensor nac[-1 -2;-3]:=nc[-1,1]*nar[1,-2,-3]

        state[i]=nal
        state[i+1]=nac

        if(i!=(length(state)-1))
            (state[i+1],convhist) = exponentiate(x->ac_prime(x,i+1,state,pars),1im*timestep/2,nac,Lanczos())
        end

    end

    #right to left

    for i in length(state):-1:2
        @tensor ac2[-1 -2;-3 -4]:=state[i-1][-1,-2,1]*state[i][1,-3,-4]

        (nac2,convhist) = exponentiate(x->ac2_prime(x,i-1,state,pars),-1im*timestep/2,ac2,Lanczos())

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme)
        @tensor nac[-1 -2;-3]:=nal[-1,-2,1]*nc[1,-3]

        state[i-1]=nac
        state[i]=permute(nar,(1,2),(3,))

        if(i!=2)
            (state[i-1],convhist) = exponentiate(x->ac_prime(x,i-1,state,pars),1im*timestep/2,nac,Lanczos())
        end
    end

    return state,pars
end

function timestep(state::FiniteMpo, H::ComAct, timestep::Number,alg::Tdvp2,pars=params(state,H))

    #left to right
    for i in 1:(length(state)-1)
        @tensor ac2[-1 -2 -3;-4 -5 -6]:=state[i][-1,-2,1,-6]*state[i+1][1,-3,-4,-5]

        (nac2,convhist) = exponentiate(x->ac2_prime(x,i,state,pars),-1im*timestep/2,ac2,Lanczos())

        (nal,nc,nar) = tsvd(permute(nac2,(1,2,6),(3,4,5)),trunc=alg.trscheme)
        @tensor nac[-1 -2;-3 -4]:=nc[-1,1]*nar[1,-2,-3,-4]

        state[i]=permute(nal,(1,2),(4,3))
        state[i+1]=nac

        if(i!=(length(state)-1))
            (state[i+1],convhist) = exponentiate(x->ac_prime(x,i+1,state,pars),1im*timestep/2,nac,Lanczos())
        end

    end

    #right to left

    for i in length(state):-1:2
        @tensor ac2[-1 -2 -3;-4 -5 -6]:=state[i-1][-1,-2,1,-6]*state[i][1,-3,-4,-5]

        (nac2,convhist) = exponentiate(x->ac2_prime(x,i-1,state,pars),-1im*timestep/2,ac2,Lanczos())

        (nal,nc,nar) = tsvd(permute(nac2,(1,2,6),(3,4,5)),trunc=alg.trscheme)

        @tensor nac[-1 -2;-3 -4]:=nal[-1,-2,-4,1]*nc[1,-3]

        state[i-1]=nac
        state[i]=permute(nar,(1,2),(3,4))

        if(i!=2)
            (state[i-1],convhist) = exponentiate(x->ac_prime(x,i-1,state,pars),1im*timestep/2,nac,Lanczos())
        end
    end

    return state,pars
end
