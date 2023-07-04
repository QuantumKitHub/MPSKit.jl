
function _update_leftEnv!(nleft::InfiniteMPS,WindowEnv::Window{E,F,E}) where {E <: Cache, F <: Cache}

    l = leftenv(WindowEnv.left,1,nleft);
    WindowEnv.middle.ldependencies[:] = similar.(WindowEnv.middle.ldependencies); # forget the old left dependencies - this forces recalculation whenever leftenv is called
    WindowEnv.middle.leftenvs[1] = l;

    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end

function _update_leftEnv!(nleft::InfiniteMPS,WindowEnv::Window{E,F,E}) where {E <: MultipleEnvironments, F <: MultipleEnvironments}
    @assert length(WindowEnv.middle) == length(WindowEnv.left)

    for (subEnvLeft,subEnvMiddle) in zip(WindowEnv.left, WindowEnv.middle) 
        l = leftenv(subEnvLeft,1,nleft);
        subEnvMiddle.ldependencies[:] = similar.(subEnvMiddle.ldependencies); # forget the old left dependencies - this forces recalculation whenever leftenv is called
        subEnvMiddle.leftenvs[1] = l;
    end
    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end


function _update_rightEnv!(nright::InfiniteMPS,WindowEnv::Window{E,F,E}) where {E <: Cache, F <: Cache}

    r = rightenv(WindowEnv.right,length(nright),nright);
    WindowEnv.middle.rdependencies[:] = similar.(WindowEnv.middle.rdependencies); # forget the old right dependencies - this forces recalculation
    WindowEnv.middle.rightenvs[end] = r;
        
    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end

function _update_rightEnv!(nright::InfiniteMPS,WindowEnv::Window{E,F,E}) where {E <: MultipleEnvironments, F <: MultipleEnvironments}
    @assert length(WindowEnv.middle) == length(WindowEnv.right)

    for (subEnvMiddle,subEnvRight) in zip(WindowEnv.middle,WindowEnv.right) # we force the windowed envs to be reculculated 
        r = rightenv(subEnvRight,length(nright),nright);
        subEnvMiddle.rdependencies[:] = similar.(subEnvMiddle.rdependencies); # forget the old right dependencies - this forces recalculation
        subEnvMiddle.rightenvs[end] = r;
        
    end
    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end


function timestep!(Ψ::WindowMPS, H::Window, t::Number, dt::Number,alg::TDVP,env::Window=environments(Ψ,H))
   
    #first evolve left state
    if !isnothing(H.left)
        nleft, _ = timestep(Ψ.left_gs, H.left, t, dt, alg, env.left; leftorthflag = true) #env gets updated in place
        _update_leftEnv!(nleft, env)
    else
        nleft = Ψ.left_gs #no need to copy, WindowMPS constructor copies automatically
    end

    # some Notes
    # - at what time do we evaluate h_ac and c? at t, t+dt/4 ? do we take both at the same time?

    #left to right sweep on window
    for i in 1:(length(Ψ)-1)
        h_ac = ∂∂AC(i,Ψ,H.middle,env.middle);
        Ψ.AC[i], converged, convhist = integrate(h_ac,Ψ.AC[i],t,-1im,dt/2,alg.integrator)
        converged == 0 &&
                @info "time evolving ac($i) failed $(convhist.normres)"

        h_c = ∂∂C(i,Ψ,H.middle,env.middle);
        Ψ.CR[i], converged, convhist = integrate(h_c,Ψ.CR[i],t,-1im,-dt/2,alg.integrator)
        converged == 0 &&
                @info "time evolving c($i) failed $(convhist.normres)"
    end

    h_ac = ∂∂AC(length(Ψ),Ψ,H.middle,env.middle);
    Ψ.AC[end], converged, convhist = integrate(h_ac,Ψ.AC[end],t,-1im,dt/2,alg.integrator)
    converged == 0 &&
            @info "time evolving ac($(length(Ψ))) failed $(convhist.normres)"

    if !isnothing(H.right)
        nright, _ = timestep(Ψ.right_gs, H.right, t, dt, alg, env.right, leftorthflag = false) #env gets updated in place
        _update_rightEnv!(nright, env)
    else
        nright = Ψ.right_gs #no need to copy, WindowMPS constructor copies automatically
    end

    #right to left sweep on window
    for i in length(Ψ):-1:2
        h_ac = ∂∂AC(i,Ψ,H.middle,env.middle);
        Ψ.AC[i], converged, convhist = integrate(h_ac,Ψ.AC[i],t+dt/2,-1im,dt/2,alg.integrator)
        converged == 0 &&
            @info "time evolving ac($i) failed $(convhist.normres)"

        h_c = ∂∂C(i-1,Ψ,H.middle,env.middle);
        Ψ.CR[i-1], converged, convhist = integrate(h_c,Ψ.CR[i-1],t+dt/2,-1im,-dt/2,alg.integrator)
        converged == 0 &&
            @info "time evolving c($(i-1)) failed $(convhist.normres)"
    end

    h_ac = ∂∂AC(1,Ψ,H.middle,env.middle);
    Ψ.AC[1], converged, convhist = integrate(h_ac,Ψ.AC[1],t+dt/2,-1im,dt/2,alg.integrator)
    converged == 0 &&
            @info "time evolving ac(1) failed $(convhist.normres)"

    return WindowMPS(nleft,Ψ.window,nright),env
end