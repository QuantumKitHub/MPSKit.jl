"onesite tdvp"
@with_kw struct Tdvp <: Algorithm
    tol::Float64 = Defaults.tol
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
end

"""
    function timestep(psi, operator, dt, alg,parameters = params(psi,operator))

time evolves psi by timestep dt using algorithm alg
"""
function timestep(state::InfiniteMPS, H::Hamiltonian, timestep::Number,alg::Tdvp,parameters::Cache=params(state,H))

    newAs = similar(state.AL)

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
        QAc,_ = leftorth!(newAcenter,alg=TensorKit.QRpos())
        Qc,_ = leftorth!(newCenter,alg=TensorKit.QRpos())
        @tensor Aleft[-1 -2;-3]:=QAc[-1,-2,1]*conj(Qc[-3,1])

        newAs[loc]     = Aleft
    end

    return InfiniteMPS(newAs; tol = alg.tolgauge, maxiter = alg.maxiter,leftgauged = true),parameters
end

function timestep(state::Union{FiniteMPS,MPSComoving}, H::Operator, timestep::Number,alg::Tdvp,pars=params(state,H))
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

"twosite tdvp (works for finite mps's)"
@with_kw struct Tdvp2 <: Algorithm
    tol::Float64 = Defaults.tol
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    trscheme = truncerr(1e-3)
end

#twosite tdvp for finite mps
function timestep(state::Union{FiniteMPS,MPSComoving}, H::Operator, timestep::Number,alg::Tdvp2,pars=params(state,H);rightorthed=false)
    #left to right
    for i in 1:(length(state)-1)
        ac2 = _permute_front(state.AC[i])*_permute_tail(state.AR[i+1])

        (nac2,convhist) = let state=state,pars=pars
            exponentiate(x->ac2_prime(x,i,state,pars),-1im*timestep/2,ac2,Lanczos())
        end

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme)

        state.AC[i] = (nal,complex(nc))
        state.AR[i+1] = _permute_front(nar)

        if(i!=(length(state)-1))
            (state.AC[i+1],convhist) = let state=state,pars=pars
                exponentiate(x->ac_prime(x,i+1,state,pars),1im*timestep/2,state.AC[i+1],Lanczos())
            end
        end

    end

    #right to left

    for i in length(state):-1:2
        ac2 = _permute_front(state.AC[i-1])*_permute_tail(state.AR[i])

        (nac2,convhist) = let state=state,pars=pars
            exponentiate(x->ac2_prime(x,i-1,state,pars),-1im*timestep/2,ac2,Lanczos())
        end

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme)

        state.AL[i-1] = nal;
        state.AC[i] = (complex(nc),_permute_front(nar));

        if(i!=2)
            (state.AC[i-1],convhist) = let state=state,pars=pars
                exponentiate(x->ac_prime(x,i-1,state,pars),1im*timestep/2,state.AC[i-1],Lanczos())
            end
        end
    end

    return state,pars
end
