# make a struct for WindowTDVP, finalize for bondexpansion and glue after that

function timestep(Ψ::WindowMPS{A,B,VL,VR},H::Union{Window,LazySum},t::Number,dt::Number,alg::TDVP,env::Cache=environments(Ψ, H)) where {A,B,VL,VR}
    #first evolve left state
    if VL === WINDOW_VARIABLE
        println("Doing left")
        nleft, _ = timestep(Ψ.left, H.left, t, dt, alg, env.left; leftorthflag=true) #env gets updated in place
        Ψ = WindowMPS(nleft,Ψ.middle,Ψ.right)
    end

    # left to right sweep on window
    for i in 1:(length(Ψ) - 1)
        h_ac = ∂∂AC(i, Ψ, H.middle, env)
        Ψ.AC[i] = integrate(h_ac, Ψ.AC[i], t, dt / 2, alg.integrator)

        h_c = ∂∂C(i, Ψ, H.middle, env)
        Ψ.CR[i] = integrate(h_c, Ψ.CR[i], t, -dt / 2, alg.integrator)
    end

    h_ac = ∂∂AC(length(Ψ), Ψ, H.middle, env)
    Ψ.AC[end] = integrate(h_ac, Ψ.AC[end], t, dt / 2, alg.integrator)

    #then evolve right state
    if VR === WINDOW_VARIABLE
        println("Doing right")
        nright, _ = timestep(Ψ.right, H.right, t + dt, dt, alg, env.right; leftorthflag=false) # env gets updated in place
        Ψ = WindowMPS(Ψ.left,Ψ.middle,nright)
    end

    # right to left sweep on window
    for i in length(Ψ):-1:2
        h_ac = ∂∂AC(i, Ψ, H.middle, env)
        Ψ.AC[i] = integrate(h_ac, Ψ.AC[i], t + dt / 2, dt / 2, alg.integrator)

        h_c = ∂∂C(i - 1, Ψ, H.middle, env)
        Ψ.CR[i - 1] = integrate(h_c, Ψ.CR[i - 1], t + dt / 2, -dt / 2, alg.integrator)
    end

    h_ac = ∂∂AC(1, Ψ, H.middle, env)
    Ψ.AC[1] = integrate(h_ac, Ψ.AC[1], t, dt / 2, alg.integrator)

    return Ψ, env
end

#=

function timestep!(
    Ψ::WindowMPS{A,B,VL,VR},
    H,
    t::Number,
    dt::Number,
    alg::TDVP,
    env=environments(Ψ, H);
) where {A,B,VL,VR}
    #first evolve left state
    if VL === WINDOW_VARIABLE
        nleft, _ = timestep(Ψ.left, H.left, t, dt, alg, env.left; leftorthflag=true) #env gets updated in place
        Ψ = WindowMPS(nleft,Ψ.middle,Ψ.right)
    end

    # left to right sweep on window
    for i in 1:(length(Ψ) - 1)
        h_ac = ∂∂AC(i, Ψ, H.middle, env.middle)
        Ψ.AC[i] = integrate(h_ac, Ψ.AC[i], t, dt / 2, alg.integrator)

        h_c = ∂∂C(i, Ψ, H.middle, env.middle)
        Ψ.CR[i] = integrate(h_c, Ψ.CR[i], t, -dt / 2, alg.integrator)
    end

    h_ac = ∂∂AC(length(Ψ), Ψ, H.middle, env.middle)
    Ψ.AC[end] = integrate(h_ac, Ψ.AC[end], t, dt / 2, alg.integrator)

    if VR === WINDOW_VARIABLE
        nright, _ = timestep(Ψ.right_gs, H.right, t + dt, dt, alg, env.right; leftorthflag=false) # env gets updated in place
        Ψ = WindowMPS(Ψ.left,Ψ.middle,nright)
    end

    # right to left sweep on window
    for i in length(Ψ):-1:2
        h_ac = ∂∂AC(i, Ψ, H.middle, env.middle)
        Ψ.AC[i] = integrate(h_ac, Ψ.AC[i], t + dt / 2, dt / 2, alg.integrator)

        h_c = ∂∂C(i - 1, Ψ, H.middle, env.middle)
        Ψ.CR[i - 1] = integrate(h_c, Ψ.CR[i - 1], t + dt / 2, -dt / 2, alg.integrator)
    end

    h_ac = ∂∂AC(1, Ψ, H.middle, env.middle)
    Ψ.AC[1] = integrate(h_ac, Ψ.AC[1], t, dt / 2, alg.integrator)

    return Ψ, env
end
=#

#=
function timestep!(
    Ψ::WindowMPS,
    H::Window,
    t::Number,
    dt::Number,
    alg::TDVP2,
    env::Window=environments(Ψ, H);
    leftevolve=false,
    rightevolve=false,
    kwargs...,
)
    singleTDVPalg = TDVP(;
        integrator=alg.integrator, tolgauge=alg.tolgauge, maxiter=alg.maxiter
    )

    # first evolve left state
    if leftevolve
        # expand the bond dimension using changebonds
        nleft, _ = leftexpand(Ψ, H.left(t), env.left; kwargs...)
        # fill it by doing regular TDVP
        nleft, _ = timestep(
            nleft, H.left, t, dt, singleTDVPalg, env.left; leftorthflag=true
        )
        _update_leftEnv!(nleft, env)
    else
        nleft = Ψ.left
    end

    # left to right sweep on window
    for i in 1:(length(Ψ) - 1)
        h_ac2 = ∂∂AC2(i, Ψ, H.middle, env.middle)
        ac2 = Ψ.AC[i] * _transpose_tail(Ψ.AR[i + 1])
        ac2 = integrate(h_ac2, ac2, t, dt / 2, alg.integrator)

        U, S, V, = tsvd!(ac2; alg=TensorKit.SVD(), trunc=alg.trscheme)

        Ψ.AC[i] = (U, S)
        Ψ.AC[i + 1] = (S, _transpose_front(V))

        if i < length(Ψ) - 1
            h_ac = ∂∂AC(i + 1, Ψ, H.middle, env.middle)
            Ψ.AC[i + 1] = integrate(h_ac, Ψ.AC[i + 1], t, -dt / 2, alg.integrator)
        end
    end

    if rightevolve
        # expand the bond dimension using changebonds
        nright, _ = rightexpand(Ψ, H.right(t), env.right; kwargs...)
        # fill it by doing regular TDVP
        nright, _ = timestep(
            nright, H.right, t + dt, dt, singleTDVPalg, env.right; leftorthflag=false
        ) #env gets updated in place
        _update_rightEnv!(nright, env)
    else
        nright = Ψ.right_gs
    end

    # right to left sweep on window
    for i in length(Ψ):-1:2
        h_ac2 = ∂∂AC2(i - 1, Ψ, H.middle, env.middle)
        ac2 = Ψ.AL[i - 1] * _transpose_tail(Ψ.AC[i])
        ac2 = integrate(h_ac2, ac2, t + dt / 2, dt / 2, alg.integrator)
        U, S, V, = tsvd!(ac2; alg=TensorKit.SVD(), trunc=alg.trscheme)

        Ψ.AC[i - 1] = (U, S)
        Ψ.AC[i] = (S, _transpose_front(V))

        if i > 2
            h_ac = ∂∂AC(i - 1, Ψ, H.middle, env.middle)
            Ψ.AC[i - 1] = integrate(h_ac, Ψ.AC[i - 1], t + dt / 2, -dt / 2, alg.integrator)
        end
    end

    return WindowMPS(nleft, Ψ.window, nright), env
end
=#
