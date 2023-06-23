"""
    timestep(Ψ, H, dt, algorithm, environments)
    timestep!(Ψ, H, dt, algorithm, environments)

Compute the time-evolved state ``Ψ′ ≈ exp(-iHdt) Ψ``.

# Arguments
- `Ψ::AbstractMPS`: current state
- `H::AbstractMPO`: evolution operator
- `dt::Number`: timestep
- `algorithm`: evolution algorithm
- `[environments]`: environment manager
"""
function timestep end, function timestep! end

"""
    TDVP{A} <: Algorithm

Single site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `expalg::A`: exponentiator algorithm
- `tolgauge::Float64`: tolerance for gauging algorithm
- `maxiter::Int`: maximum amount of gauging iterations
"""
@kwdef struct TDVP{A} <: Algorithm
    expalg::A = Lanczos(; tol=Defaults.tol)
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
end

function timestep(state::InfiniteMPS, H, time::Number, timestep::Number, alg::TDVP, envs::Union{Cache,MultipleEnvironments}=environments(state,H); leftorthflag=true)
    #=
    can we understand whether this is actually correct up to some order? what order?
    =#
    temp_ACs = similar(state.AC);
    temp_CRs = similar(state.CR);

    @sync for (loc,(ac,c)) in enumerate(zip(state.AC,state.CR))
        @Threads.spawn begin
            h_ac = MPSKit.∂∂AC($loc,$state,$H,$envs);
            $temp_ACs[loc], converged, convhist = integrate(h_ac,$ac,$time,-1im,$timestep,alg.expalg)
            converged == 0 &&
                @info "time evolving ac($loc) failed $(convhist.normres)"
        end

        @Threads.spawn begin
            h_c = MPSKit.∂∂C($loc,$state,$H,$envs);
            $temp_CRs[loc], converged, convhist = integrate(h_c,$c,$time,-1im,$timestep,alg.expalg)
            converged == 0 &&
                @info "time evolving ac($loc) failed $(convhist.normres)"
        end
    end

    if leftorthflag

        for loc in 1:length(state)
            #find AL that best fits these new Acenter and centers
            QAc,_ = leftorth!(temp_ACs[loc],alg=TensorKit.QRpos())
            Qc,_ = leftorth!(temp_CRs[loc],alg=TensorKit.QRpos())
            @plansor temp_ACs[loc][-1 -2;-3] = QAc[-1 -2;1]*conj(Qc[-3;1])
        end
        nstate = InfiniteMPS(temp_ACs,state.CR[end]; tol = alg.tolgauge, maxiter = alg.maxiter)
    
    else

        for loc in 1:length(state)
            #find AR that best fits these new Acenter and centers
            _,QAc = rightorth!(_transpose_tail(temp_ACs[loc]),alg=TensorKit.LQpos())
            _,Qc = rightorth!(temp_CRs[mod1(loc-1,end)],alg=TensorKit.LQpos())
            temp_ACs[loc] = _transpose_front(Qc'*QAc)
        end
        nstate = InfiniteMPS(state.CR[0],temp_ACs; tol = alg.tolgauge, maxiter = alg.maxiter)
    end
    
    recalculate!(envs,nstate)
    nstate,envs
end

#should also have timestep for FiniteMPS/WindowMPS with H::Union{TimedOperator,SumOfOperators}

function _update_leftEnv!(nleft::InfiniteMPS,WindowEnv::Window{O}) where O <: Cache

    for (subEnvLeft,subEnvMiddle) in zip(WindowEnv.left, WindowEnv.middle)  # we force the windowed envs to be recalculated 
        l = leftenv(subEnvLeft,1,nleft);
        subEnvMiddle.ldependencies[:] = similar.(subEnvMiddle.ldependencies); # forget the old left dependencies - this forces recalculation whenever leftenv is called
        subEnvMiddle.leftenvs[1] = l;
    end
    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end

function _update_rightEnv!(nright::InfiniteMPS,WindowEnv::Window{O}) where O <: Cache

    for (subEnvMiddle,subEnvRight) in zip(WindowEnv.middle,WindowEnv.right) # we force the windowed envs to be reculculated 
        r = rightenv(subEnvRight,length(state),nright);
        subEnvMiddle.rdependencies[:] = similar.(subEnvMiddle.rdependencies); # forget the old right dependencies - this forces recalculation
        subEnvMiddle.rightenvs[end] = r;
        
    end
    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end

function timestep!(state::WindowMPS,H::Window, t::Number, dt::Number,alg::TDVP,env::Window{C,C,C}=environments(state,H)) where C <: Union{Cache,MultipleEnvironments}

    #first evolve left state
    if !isnothing(H.left)
        nleft, _ = timestep(state.left_gs, H.left, t, dt, alg, env.left) #env gets updated in place, check this to be sure
        env = _update_leftEnv!(nleft, env)
    end

    # some Notes
    # - at what time do we evaluate h_ac and c? at t, t+dt/4 ? do we take both at the same time?

    #left to right sweep on window
    for i in 1:(length(state)-1)
        h_ac = ∂∂AC(i,state,H.middle,env.middle);
        state.AC[i] = integrate(h_ac,state.AC[i],t,-1im,dt/2,alg.integrator)

        h_c = ∂∂C(i,state,H.middle,env.middle);
        state.CR[i] = integrate(h_c,state.CR[i],t,-1im,-dt/2,alg.integrator)
    end

    h_ac = ∂∂AC(length(state),state,H.middle,env.middle);
    state.AC[end] = integrate(h_ac,state.AC[end],t,-1im,dt/2,alg.integrator)

    if !isnothing(H.right)
        nright, _ = timestep(state.right_gs, H.right, t, dt, alg, env.right) #env gets updated in place, check this to be sure
        env = _update_rightEnv!(nright, env)
    end

    #right to left sweep on window
    for i in length(state):-1:2
        h_ac = ∂∂AC(i,state,H.middle,env.middle);
        state.AC[i] = integrate(h_ac,state.AC[i],t+dt/2,-1im,dt/2,alg.integrator)

        h_c = ∂∂C(i-1,state,H.middle,env.middle);
        state.CR[i-1] = integrate(h_c,state.CR[i-1],t+dt/2,-1im,-dt/2,alg.integrator)
    end

    h_ac = ∂∂AC(1,state,H.middle,env.middle);
    state.AC[1] = integrate(h_ac,state.AC[1],t+dt/2,-1im,dt/2,alg.integrator)

    return WindowMPS(nleft,state.window,nright),env
end

"""
    TDVP2{A} <: Algorithm

2-site [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601)
algorithm for time evolution.

# Fields
- `intalg::A`: integrator algorithm (defaults to Lanczos exponentiation)
- `tolgauge::Float64`: tolerance for gauging algorithm
- `maxiter::Int`: maximum amount of gauging iterations
- `trscheme`: truncation algorithm for [tsvd][TensorKit.tsvd](@ref)
"""
@kwdef struct TDVP2{A} <: Algorithm
    intalg::A = Lanczos(; tol=Defaults.tol)
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    trscheme = truncerr(1e-3)
end

timestep(state, H, dt, alg, env) = timestep(state, H, 0., dt, alg, env) 