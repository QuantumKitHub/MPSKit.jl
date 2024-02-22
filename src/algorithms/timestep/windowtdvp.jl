"""
    WindowTDVP{A} <: Algorithm

[Mixed TDVP](https://arxiv.org/abs/2007.15035) algorithm for time evolution.

# Fields
- `finite_alg::A`: algorithm to do the timestep of the finite part of the WindowMPS. This can be `TDVP2` to expand the bonddimension.
- `infinite_alg::A`: algorithm to do the timestep of the infinite part of the WindowMPS
- `finalize::F`: user-supplied function which is applied after each timestep, with
    signature `finalize(t, Ψ, H, envs) -> Ψ, envs`. Can be used to enlarge the bond dimension of the infinite part.
"""
@kwdef struct WindowTDVP{A,B,F} <: Algorithm
    finite_alg::A = TDVP()
    infinite_alg::B = TDVP()
    finalize::F = Defaults._finalize
end

function timestep!(Ψ::WindowMPS{A,B,VL,VR},H::Union{Window,LazySum{<:Window}},t::Number,dt::Number,alg::WindowTDVP,env=environments(Ψ, H)) where {A,B,VL,VR}
    
    #first evolve left state
    if VL === WINDOW_VARIABLE
        nleft, _ = timestep(Ψ.left, H.left, t, dt, alg.infinite_alg, env.left; leftorthflag=true) #env gets updated in place
        Ψ = WindowMPS(nleft,Ψ.middle,Ψ.right)
    end

    Ψ, env = ltr_sweep!(Ψ, H.middle, t, dt / 2, alg.finite_alg, env)

    #then evolve right state
    if VR === WINDOW_VARIABLE
        nright, _ = timestep(Ψ.right, H.right, t + dt, dt, alg.infinite_alg, env.right; leftorthflag=false) # env gets updated in place
        Ψ = WindowMPS(Ψ.left,Ψ.middle,nright)
    end

    Ψ, env = rtl_sweep!(Ψ, H.middle, t + dt / 2, dt / 2, alg.finite_alg, env)
    
    return Ψ, env
end