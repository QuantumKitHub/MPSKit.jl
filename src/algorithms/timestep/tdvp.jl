"""
    timestep(Ψ, H, t, dt, algorithm, environments)
    timestep!(Ψ, H, t, dt, algorithm, environments)

Compute the time-evolved state ``Ψ′ ≈ exp(-iHdt) Ψ`` where H can be time-dependent. For a time-independent H (i.e. not a TimedOperator) t is ignored.

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

function timestep(Ψ::InfiniteMPS, H, time::Number, timestep::Number, alg::TDVP, 
    envs::Union{Cache,MultipleEnvironments}=environments(Ψ,H); leftorthflag=true)
    #=
    can we understand whether this is actually correct up to some order? what order?
    =#
    temp_ACs = similar(Ψ.AC);
    temp_CRs = similar(Ψ.CR);

    @sync for (loc,(ac,c)) in enumerate(zip(Ψ.AC,Ψ.CR))
        @Threads.spawn begin
            h_ac = MPSKit.∂∂AC($loc,$Ψ,$H,$envs);
            $temp_ACs[loc], converged, convhist = integrate(h_ac,$ac,$time,-1im,$timestep,alg.expalg)
            converged == 0 &&
                @info "time evolving ac($loc) failed $(convhist.normres)"
        end

        @Threads.spawn begin
            h_c = MPSKit.∂∂C($loc,$Ψ,$H,$envs);
            $temp_CRs[loc], converged, convhist = integrate(h_c,$c,$time,-1im,$timestep,alg.expalg)
            converged == 0 &&
                @info "time evolving ac($loc) failed $(convhist.normres)"
        end
    end

    if leftorthflag

        for loc in 1:length(Ψ)
            #find AL that best fits these new Acenter and centers
            QAc,_ = leftorth!(temp_ACs[loc],alg=TensorKit.QRpos())
            Qc,_ = leftorth!(temp_CRs[loc],alg=TensorKit.QRpos())
            @plansor temp_ACs[loc][-1 -2;-3] = QAc[-1 -2;1]*conj(Qc[-3;1])
        end
        newΨ = InfiniteMPS(temp_ACs,Ψ.CR[end]; tol = alg.tolgauge, maxiter = alg.maxiter)
    
    else

        for loc in 1:length(Ψ)
            #find AR that best fits these new Acenter and centers
            _,QAc = rightorth!(_transpose_tail(temp_ACs[loc]),alg=TensorKit.LQpos())
            _,Qc = rightorth!(temp_CRs[mod1(loc-1,end)],alg=TensorKit.LQpos())
            temp_ACs[loc] = _transpose_front(Qc'*QAc)
        end
        newΨ = InfiniteMPS(Ψ.CR[0],temp_ACs; tol = alg.tolgauge, maxiter = alg.maxiter)
    end
    
    recalculate!(envs,newΨ)
    newΨ,envs
end

function timestep!(Ψ::FiniteMPS, H, t::Number, dt::Number, alg::TDVP, 
    envs::Union{Cache,MultipleEnvironments}=environments(Ψ, H))

    for i in 1:(length(Ψ) - 1)
        h_ac = ∂∂AC(i, Ψ, H, envs)
        Ψ.AC[i], converged, convhist = integrate(h_ac, Ψ.AC[i], t, -1im, dt / 2, alg.integrator)
        converged == 0 &&
                @info "time evolving ac($i) on the left->right sweep failed $(convhist.normres)"

        h_c = ∂∂C(i, Ψ, H, envs)
        Ψ.CR[i], converged, convhist = integrate(h_c, Ψ.CR[i], t, 1im, dt / 2, alg.integrator)
        converged == 0 &&
                @info "time evolving c($i) on the left->right sweep failed $(convhist.normres)"
    end

    h_ac = ∂∂AC(length(Ψ), Ψ, H, envs)
    Ψ.AC[end], converged, convhist = integrate(h_ac, Ψ.AC[end], t, -1im, dt / 2, alg.integrator)
    converged == 0 &&
                @info "time evolving ac($(length(Ψ))) on the left->right sweep failed $(convhist.normres)"

    for i in length(Ψ):-1:2
        h_ac = ∂∂AC(i, Ψ, H, envs)
        Ψ.AC[i], converged, convhist = integrate(h_ac, Ψ.AC[i], t + dt / 2, -1im, dt / 2, alg.integrator)
        converged == 0 &&
                @info "time evolving c($i) on the right->left sweep failed $(convhist.normres)"

        h_c = ∂∂C(i - 1, Ψ, H, envs)
        Ψ.CR[i - 1], converged, convhist = integrate(h_c, Ψ.CR[i - 1], t + dt / 2, 1im, dt / 2, alg.integrator)
        converged == 0 &&
                @info "time evolving c($i) on the right->left sweep failed $(convhist.normres)"
    end

    h_ac = ∂∂AC(1, Ψ, H, envs)
    Ψ.AC[1], converged, convhist = integrate(h_ac, Ψ.AC[1], t + dt / 2, -1im, dt / 2, alg.integrator)
    converged == 0 &&
        @info "time evolving ac(1) on the right->left sweep failed $(convhist.normres)"
    return Ψ, envs
end

function _update_leftEnv!(nleft::InfiniteMPS,WindowEnv::Window{O,O,O}) where O <: Cache

    l = leftenv(WindowEnv.left,1,nleft);
    WindowEnv.middle.ldependencies[:] = similar.(WindowEnv.middle.ldependencies); # forget the old left dependencies - this forces recalculation whenever leftenv is called
    WindowEnv.middle.leftenvs[1] = l;

    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end

function _update_leftEnv!(nleft::InfiniteMPS,WindowEnv::Window{O,O,O}) where O <: MultipleEnvironments
    @assert length(WindowEnv.middle) == length(WindowEnv.left)

    for (subEnvLeft,subEnvMiddle) in zip(WindowEnv.left, WindowEnv.middle) 
        l = leftenv(subEnvLeft,1,nleft);
        subEnvMiddle.ldependencies[:] = similar.(subEnvMiddle.ldependencies); # forget the old left dependencies - this forces recalculation whenever leftenv is called
        subEnvMiddle.leftenvs[1] = l;
    end
    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end


function _update_rightEnv!(nright::InfiniteMPS,WindowEnv::Window{O,O,O}) where O <: Cache

    r = rightenv(WindowEnv.right,length(nright),nright);
    WindowEnv.middle.rdependencies[:] = similar.(WindowEnv.middle.rdependencies); # forget the old right dependencies - this forces recalculation
    WindowEnv.middle.rightenvs[end] = r;
        
    Window(WindowEnv.left,WindowEnv.middle,WindowEnv.right )
end

function _update_rightEnv!(nright::InfiniteMPS,WindowEnv::Window{O,O,O}) where O <: MultipleEnvironments
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
        nleft, _ = timestep(Ψ.left_gs, H.left, t, dt, alg, env.left) #env gets updated in place, check this to be sure
        env = _update_leftEnv!(nleft, env)
    end

    # some Notes
    # - at what time do we evaluate h_ac and c? at t, t+dt/4 ? do we take both at the same time?

    #left to right sweep on window
    for i in 1:(length(Ψ)-1)
        h_ac = ∂∂AC(i,Ψ,H.middle,env.middle);
        Ψ.AC[i] = integrate(h_ac,Ψ.AC[i],t,-1im,dt/2,alg.integrator)

        h_c = ∂∂C(i,Ψ,H.middle,env.middle);
        Ψ.CR[i] = integrate(h_c,Ψ.CR[i],t,-1im,-dt/2,alg.integrator)
    end

    h_ac = ∂∂AC(length(Ψ),Ψ,H.middle,env.middle);
    Ψ.AC[end] = integrate(h_ac,Ψ.AC[end],t,-1im,dt/2,alg.integrator)

    if !isnothing(H.right)
        nright, _ = timestep(Ψ.right_gs, H.right, t, dt, alg, env.right) #env gets updated in place, check this to be sure
        env = _update_rightEnv!(nright, env)
    end

    #right to left sweep on window
    for i in length(Ψ):-1:2
        h_ac = ∂∂AC(i,Ψ,H.middle,env.middle);
        Ψ.AC[i] = integrate(h_ac,Ψ.AC[i],t+dt/2,-1im,dt/2,alg.integrator)

        h_c = ∂∂C(i-1,Ψ,H.middle,env.middle);
        Ψ.CR[i-1] = integrate(h_c,Ψ.CR[i-1],t+dt/2,-1im,-dt/2,alg.integrator)
    end

    h_ac = ∂∂AC(1,Ψ,H.middle,env.middle);
    Ψ.AC[1] = integrate(h_ac,Ψ.AC[1],t+dt/2,-1im,dt/2,alg.integrator)

    return WindowMPS(nleft,Ψ.window,nright),env
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

function timestep!(Ψ::AbstractFiniteMPS, H, t::Number, dt::Number, alg::TDVP2,
    envs=environments(Ψ, H))
    #left to right
    for i in 1:(length(Ψ) - 1)
        ac2 = _transpose_front(Ψ.AC[i]) * _transpose_tail(Ψ.AR[i + 1])

        h_ac2 = ∂∂AC2(i, Ψ, H, envs)
        nac2, converged, convhist = integrate(h_ac2, ac2, t, -1im, dt / 2, alg.integrator)

        nal, nc, nar = tsvd(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())

        Ψ.AC[i] = (nal, complex(nc))
        Ψ.AC[i + 1] = (complex(nc), _transpose_front(nar))

        if i != (length(Ψ) - 1)
            Ψ.AC[i + 1], converged, convhist = integrate(∂∂AC(i + 1, Ψ, H, envs), Ψ.AC[i + 1], t, 1im, dt / 2,
                                        alg.integrator)
        end
    end

    #right to left
    for i in length(Ψ):-1:2
        ac2 = _transpose_front(Ψ.AL[i - 1]) * _transpose_tail(Ψ.AC[i])

        h_ac2 = ∂∂AC2(i - 1, Ψ, H, envs)
        (nac2, converged, convhist) = integrate(h_ac2, ac2, t, -1im, dt / 2, alg.integrator)

        nal, nc, nar = tsvd(nac2; trunc=alg.trscheme, alg=TensorKit.SVD())

        Ψ.AC[i - 1] = (nal, complex(nc))
        Ψ.AC[i] = (complex(nc), _transpose_front(nar))

        if i != 2
            Ψ.AC[i - 1], converged, convhist = integrate(∂∂AC(i - 1, Ψ, H, envs), Ψ.AC[i - 1], t, 1im, dt / 2,
                                        alg.integrator)
        end
    end

    return Ψ, envs
end

# time-independent version
timestep(Ψ, H, dt, alg, env) = timestep(Ψ, H, 0., dt, alg, env) 

#copying version
function timestep(Ψ::AbstractFiniteMPS, H, time ,timestep, alg::Union{TDVP,TDVP2},
    envs=environments(Ψ, H))
return timestep!(copy(Ψ), H, time, timestep, alg, envs)
end

#given that I intend to put my infinite sweep algorithm into MPSKit, we might consider an abstract TDVP algorithm?