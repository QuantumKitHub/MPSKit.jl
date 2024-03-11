```@meta
EditURL = "../../../../../examples/quantum1d/3.ising-dqpt/main.jl"
```

[![](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maartenvd/MPSKit.jl/gh-pages?filepath=dev/examples/quantum1d/3.ising-dqpt/main.ipynb)
[![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](https://nbviewer.jupyter.org/github/maartenvd/MPSKit.jl/blob/gh-pages/dev/examples/quantum1d/3.ising-dqpt/main.ipynb)
[![](https://img.shields.io/badge/download-project-orange)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/maartenvd/MPSKit.jl/examples/tree/gh-pages/dev/examples/quantum1d/3.ising-dqpt)

# DQPT in the Ising model(@id demo_dqpt)

In this tutorial we will try to reproduce the results from
[this paper](https://arxiv.org/pdf/1206.2505.pdf). The needed packages are

````julia
using MPSKit, MPSKitModels, TensorKit
````

Dynamical quantum phase transitions (DQPT in short) are signatures of equilibrium phase transitions in a dynamical quantity - the loschmidth echo.
This quantity is given by ``L(t) = \frac{-2}{N} ln(| < \psi(t) | \psi(0) > |) `` where ``N`` is the system size.
One typically starts from a groundstate and then quenches the hamiltonian to a different point.
Non analycities in the loschmidth echo are called 'dynamical quantum phase transitions'.

In the mentioned paper they work with

``H(g) = - \sum^{N-1}_{i=1} \sigma^z_i \sigma^z_{i+1} + g \sum_{i=1}^N \sigma^x_i``

and show that divergences occur when quenching across the critical point (g₀ → g₁) for ``t^*_n = t^*(n+\frac{1}{2})`` with ``t^* = \pi/e(g_1,k^*)``, ``cos(k^*) = (1+g_0 g_1) / (g_0 + g_1)``, `` e(g,k) = \sqrt{(g-cos k)^2 + sin^2 k}``.

The outline of the tutorial is as follows. We will pick ``g₀ = 0.5``, ``g₁ = 2.0``, and perform the time evolution at different system sizes and compare with the thermodynamic limit.
For those ``g`` we expect non-analicities to occur at ``t_n ≈ 2.35 (n + 1/2)``.

First we construct the hamiltonian in mpo form, and obtain the pre-quenched groundstate:

````julia
H₀ = transverse_field_ising(; g=-0.5)

L = 20
ψ₀ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^10)
ψ₀, _ = find_groundstate(ψ₀, H₀, DMRG());
````

````
[ Info: DMRG init:	obj = +9.914551645185e+00	err = 1.3697e-01
[ Info: DMRG   1:	obj = -2.040021698083e+01	err = 9.7284195077e-04	time = 0.44 sec
[ Info: DMRG   2:	obj = -2.040021786696e+01	err = 7.5203822885e-06	time = 0.35 sec
[ Info: DMRG   3:	obj = -2.040021786703e+01	err = 9.3598556558e-07	time = 0.09 sec
[ Info: DMRG   4:	obj = -2.040021786703e+01	err = 8.3709447459e-10	time = 0.02 sec
[ Info: DMRG conv 5:	obj = -2.040021786703e+01	err = 6.6974743335e-13	time = 0.91 sec

````

## Finite MPS quenching

We can define a helper function that measures the loschmith echo

````julia
echo(ψ₀::FiniteMPS, ψₜ::FiniteMPS) = -2 * log(abs(dot(ψ₀, ψₜ))) / length(ψ₀)
@assert isapprox(echo(ψ₀, ψ₀), 0, atol=1e-10)
````

We will initially use a two-site TDVP scheme to dynamically increase the bond dimension while time evolving, and later on switch to a faster one-site scheme. A single timestep can be done using

````julia
H₁ = transverse_field_ising(; g=-2.0)
ψₜ = deepcopy(ψ₀)
dt = 0.01
ψₜ, envs = timestep(ψₜ, H₁, 0, dt, TDVP2(; trscheme=truncdim(20)));
````

"envs" is a kind of cache object that keeps track of all environments in `ψ`. It is often advantageous to re-use the environment, so that mpskit doesn't need to recalculate everything.

Putting it all together, we get

````julia
function finite_sim(L; dt=0.05, finaltime=5.0)
    ψ₀ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^10)
    ψ₀, _ = find_groundstate(ψ₀, H₀, DMRG())

    ψₜ = deepcopy(ψ₀)
    envs = environments(ψₜ, H₁)

    echos = [echo(ψₜ, ψ₀)]
    times = collect(0:dt:finaltime)

    for t in times[2:end]
        alg = t > 3 * dt ? TDVP() : TDVP2(; trscheme=truncdim(50))
        ψₜ, envs = timestep(ψₜ, H₁, 0, dt, alg, envs)
        push!(echos, echo(ψₜ, ψ₀))
    end

    return times, echos
end
````

````
finite_sim (generic function with 1 method)
````

![](finite_timeev.png)

## Infinite MPS quenching

Similarly we could start with an initial infinite state and find the pre-quench groundstate:

````julia
ψ₀ = InfiniteMPS([ℂ^2], [ℂ^10])
ψ₀, _ = find_groundstate(ψ₀, H₀, VUMPS());
````

````
[ Info: VUMPS init:	obj = +4.927142607127e-01	err = 3.7065e-01
[ Info: VUMPS   1:	obj = -1.062269158559e+00	err = 3.2878729027e-02	time = 1.11 sec
[ Info: VUMPS   2:	obj = -1.063544409816e+00	err = 1.0900961086e-05	time = 0.01 sec
[ Info: VUMPS   3:	obj = -1.063544409973e+00	err = 6.2659245120e-08	time = 0.01 sec
[ Info: VUMPS   4:	obj = -1.063544409973e+00	err = 9.8860768603e-10	time = 0.01 sec
[ Info: VUMPS   5:	obj = -1.063544409973e+00	err = 6.5293129644e-11	time = 0.00 sec
[ Info: VUMPS   6:	obj = -1.063544409973e+00	err = 6.4544340049e-12	time = 0.00 sec
[ Info: VUMPS conv 7:	obj = -1.063544409973e+00	err = 6.4503123241e-13	time = 1.14 sec

````

The dot product of two infinite matrix product states scales as  ``\alpha ^N`` where ``α`` is the dominant eigenvalue of the transfer matrix.
It is this ``α`` that is returned when calling

````julia
dot(ψ₀, ψ₀)
````

````
0.9999999999999993 - 8.56544100434766e-17im
````

so the loschmidth echo takes on the pleasant form

````julia
echo(ψ₀::InfiniteMPS, ψₜ::InfiniteMPS) = -2 * log(abs(dot(ψ₀, ψₜ)))
@assert isapprox(echo(ψ₀, ψ₀), 0, atol=1e-10)
````

This time we cannot use a two-site scheme to grow the bond dimension, as this isn't implemented (yet).
Instead, we have to make use of the changebonds machinery.
Multiple algorithms are available, but we will only focus on `OptimalEpand()`.
Growing the bond dimension by ``5`` can be done by calling:

````julia
ψₜ = deepcopy(ψ₀)
ψₜ, envs = changebonds(ψₜ, H₁, OptimalExpand(; trscheme=truncdim(5)));
````

a single timestep is easy

````julia
dt = 0.01
ψₜ, envs = timestep(ψₜ, H₁, 0, dt, TDVP(), envs);
````

With performance in mind we should once again try to re-use these "envs" cache objects.
The final code is

````julia
function infinite_sim(dt=0.05, finaltime=5.0)
    ψ₀ = InfiniteMPS([ℂ^2], [ℂ^10])
    ψ₀, _ = find_groundstate(ψ₀, H₀, VUMPS())

    ψₜ = deepcopy(ψ₀)
    envs = environments(ψₜ, H₁)

    echos = [echo(ψₜ, ψ₀)]
    times = collect(0:dt:finaltime)

    for t in times[2:end]
        if t < 50dt # if t is sufficiently small, we increase the bond dimension
            ψₜ, envs = changebonds(ψₜ, H₁, OptimalExpand(; trscheme=truncdim(1)), envs)
        end
        (ψₜ, envs) = timestep(ψₜ, H₁, 0, dt, TDVP(), envs)
        push!(echos, echo(ψₜ, ψ₀))
    end

    return times, echos
end
````

````
infinite_sim (generic function with 3 methods)
````

![](infinite_timeev.png)

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

