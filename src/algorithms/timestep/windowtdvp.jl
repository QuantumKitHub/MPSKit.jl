
function _update_leftEnv!(
    nleft::InfiniteMPS, WindowEnv::Window{E,F,E}
) where {E<:Cache,F<:Cache}
    l = leftenv(WindowEnv.left, 1, nleft)
    WindowEnv.middle.ldependencies[:] = similar.(WindowEnv.middle.ldependencies) # forget the old left dependencies - this forces recalculation whenever leftenv is called
    WindowEnv.middle.leftenvs[1] = l

    return Window(WindowEnv.left, WindowEnv.middle, WindowEnv.right)
end

function _update_rightEnv!(
    nright::InfiniteMPS, WindowEnv::Window{E,F,E}
) where {E<:Cache,F<:Cache}
    r = rightenv(WindowEnv.right, length(nright), nright)
    WindowEnv.middle.rdependencies[:] = similar.(WindowEnv.middle.rdependencies) # forget the old right dependencies - this forces recalculation
    WindowEnv.middle.rightenvs[end] = r

    return Window(WindowEnv.left, WindowEnv.middle, WindowEnv.right)
end

#Note: this needs to be tested
function leftexpand(
    st::WindowMPS,
    H,
    Envs;
    singval=1e-2,
    growspeed=10,
)
    (U, S, V) = tsvd(st.left_gs.CR[1]; alg=TensorKit.SVD())

    if minimum([minimum(diag(v)) for (k, v) in blocks(S)]) > singval
        (nst, _) = changebonds(
            st.left_gs, H, OptimalExpand(; trscheme=truncbelow(singval, growspeed)), Envs
        )

        # the AL-bond dimension changed, and therefore our window also needs updating
        v = TensorMap(
            rand, ComplexF64, left_virtualspace(nst, 0), left_virtualspace(st.left_gs, 0)
        )
        (vals, vecs) = eigsolve(
            flip(TransferMatrix(st.left_gs.AL, nst.AL)), v, 1, :LM, Arnoldi()
        )
        rho = pinv(nst.CR[0]) * vecs[1] * st.left_gs.CR[0] #CR[0] == CL[1]
        st.AC[1] = _transpose_front(normalize!(rho * st.CR[0]) * _transpose_tail(st.AR[1]))

        recalculate!(Envs, nst) #updates left infinite env based on expanded state
        return nst, Envs
    end
    return st.left_gs, Envs
end

function rightexpand(
    st::WindowMPS,
    H,
    Envs;
    singval=1e-2,
    growspeed=10,
)
    (U, S, V) = tsvd(st.left_gs.CR[1]; alg=TensorKit.SVD())

    if minimum([minimum(diag(v)) for (k, v) in blocks(S)]) > singval
        (nst, _) = changebonds(
            st.right_gs,
            H,
            OptimalExpand(; trscheme=truncbelow(singval, growspeed)),
            Envs,
        )

        v = TensorMap(
            rand, ComplexF64, right_virtualspace(st.right_gs, 0), right_virtualspace(nst, 0)
        )
        (vals, vecs) = eigsolve(
            TransferMatrix(st.right_gs.AR, nst.AR), v, 1, :LM, Arnoldi()
        )
        rho = st.right_gs.CR[0] * vecs[1] * pinv(nst.CR[0])
        st.AC[end] = st.AL[end] * normalize!(st.CR[end] * rho)

        recalculate!(Envs, nst) #updates right infinite env based on expanded state
        return nst, Envs
    end
    return st.right_gs, Envs
end

function timestep!(
    Ψ::WindowMPS,
    H::Window,
    t::Number,
    dt::Number,
    alg::TDVP,
    env::Window=environments(Ψ, H);
    leftevolve=true,
    rightevolve=true,
)
    #first evolve left state
    if leftevolve
        nleft, _ = timestep(Ψ.left_gs, H.left, t, dt, alg, env.left; leftorthflag=true) #env gets updated in place
        _update_leftEnv!(nleft, env)
    else
        nleft = Ψ.left_gs
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

    if rightevolve
        nright, _ = timestep(Ψ.right_gs, H.right, t + dt, dt, alg, env.right; leftorthflag=false) # env gets updated in place
        _update_rightEnv!(nright, env)
    else
        nright = Ψ.right_gs
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

    return WindowMPS(nleft, Ψ.window, nright), env
end

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
        nleft = Ψ.left_gs
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
