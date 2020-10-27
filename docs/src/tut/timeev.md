# DQPT in ising

In this tutorial we will try to reproduce the results from [this paper](https://arxiv.org/pdf/1206.2505.pdf). The needed packages are

```julia
using MPSKit,MPSKitModels,TensorKit,ProgressMeter
```

Dynamical quantum phase transitions (DQPT in short) are signatures of equilibrium phase transitions in a dynamical quantity - the loschmidth echo. This quantity is given by ``L(t) = \frac{-2}{N} ln(| < \psi(t) | \psi(0) > |) `` where N is the system size. One typically starts from a groundstate and then quenches the hamiltonian to a different point. Non analycities in the loschmidth echo are called 'dynamical quantum phase transitions'.

In the mentioned paper they work with ``H(g) = - \sum^{N-1}_{i=1} \sigma^z_i \sigma^z_{i+1} + g \sum_{i=1}^N \sigma^x_i`` and show that divergences occur when quenching across the critical point (g₀→g₁) for ``t^*_n = t^*(n+\frac{1}{2})`` with ``t^* = \pi/e(g_1,k^*)``, ``cos(k^*) = (1+g_0 g_1) / (g_0 + g_1)``, `` e(g,k) = \sqrt{(g-cos k)^2 + sin^2 k}``.

The outline of the tutorial is as follows. We will pick g₀ = 0.5, g₁ = 2.0, perform the time evolution at different system sizes and compare with the thermodynamic limit. For those g we expect non-analicities to occur at ``t_n ≈ 2.35 (n + 1/2)``.

First we construct the hamiltonian in mpo form:
```julia
function ising_ham(g)
    (σˣ,σʸ,σᶻ) = nonsym_spintensors(1//2).*2;

    data = Array{Any,3}(missing,1,3,3);
    data[1,1,1] = one(σˣ); data[1,end,end] = one(σˣ);
    data[1,1,2] = -σᶻ;
    data[1,2,end] = σᶻ;
    data[1,1,end] = g*σˣ;

    MPOHamiltonian(data);
end
```

## Finite MPS quenching

Construct an initial finite system
```julia
len = 20;
init = FiniteMPS(rand,ComplexF64,len,ℂ^2,ℂ^10);
```

Find the pre-quench groundstate
```julia
(ψ₀,_) = find_groundstate(init,ising_ham(0.5),Dmrg());
```

We can define a help function that measures the loschmith echo
```julia
echo(ψ₀::FiniteMPS,ψₜ::FiniteMPS) = -2*log(abs(dot(ψ₀,ψₜ)))/length(ψ₀)
@assert isapprox(echo(ψ₀,ψ₀),0,atol=1e-10);
```

we will initially use a 2site tdvp scheme to increase the bond dimension while time evolving, and later on switch to a faster one-site scheme. A single timestep can be done using
```julia
ψₜ = copy(ψ₀);
dt = 0.01;

(ψₜ,envs) = timestep(ψₜ,ising_ham(2),dt,Tdvp2(trscheme=truncdim(20)));
```

"envs" is a kind of cache object that keeps track of all environments in ψ. It is often advantageous to re-use the environment, so that mpskit doesn't need to recalculate everything.

Putting it all together, we get
```julia
function finite_sim(len; dt = 0.05, finaltime = 5.0)
    ψ₀ = FiniteMPS(rand,ComplexF64,len,ℂ^2,ℂ^10);
    (ψ₀,_) = find_groundstate(ψ₀,ising_ham(0.5),Dmrg());

    post_quench_ham = ising_ham(2);
    ψₜ = copy(ψ₀);
    envs = environments(ψₜ,post_quench_ham);

    echos = [echo(ψₜ,ψ₀)];
    times = collect(0:dt:finaltime);

    @showprogress for t = times[2:end]
        alg = t > 3*dt ? Tdvp() : Tdvp2(trscheme = truncdim(50))
        (ψₜ,envs) = timestep(ψₜ,post_quench_ham,dt,alg,envs);
        push!(echos,echo(ψₜ,ψ₀))
    end

    return (times,echos)
end
```
![](finite_timeev.png)

## Infinite MPS quenching

Similarly we start with an initial infinite state
```julia
init = InfiniteMPS([ℂ^2],[ℂ^10]);
```

and find the pre-quench groundstate
```julia
(ψ₀,_) = find_groundstate(init,ising_ham(0.5),Vumps());
```

The dot product of two infinite matrix product states scales as  ``\alpha ^N`` where α is the dominant eigenvalue of the transfer matrix. It is this α that is returned when calling
```julia
dot(ψ₀,ψ₀)
```
so the loschmidth echo takes on the pleasant form

```julia
echo(ψ₀::InfiniteMPS,ψₜ::InfiniteMPS) = -2*log(abs(dot(ψ₀,ψₜ)))
@assert isapprox(echo(ψ₀,ψ₀),0,atol=1e-10);
```

This time we cannot use a 2site scheme to grow the bond dimension, as this isn't implemented (yet). Instead, we have to make use of the changebonds machinery. Multiple algorithms are available, but we will only focus on OptimalEpand(). Growing the bond dimension by 5 can be done by calling:
```julia
ψₜ = copy(ψ₀);
(ψₜ,envs) = changebonds(ψₜ,ising_ham(2),OptimalExpand(trscheme=truncdim(5)));
```

a single timestep is easy
```julia
dt = 0.01;

(ψₜ,envs) = timestep(ψₜ,ising_ham(2),dt,Tdvp(),envs);
```

With performance in mind we should once again try to re-use these "envs" cache objects. The final code is

```julia
function infinite_sim(dt = 0.05, finaltime = 5.0)
    ψ₀ = InfiniteMPS([ℂ^2],[ℂ^10]);
    (ψ₀,_) = find_groundstate(ψ₀,ising_ham(0.5),Vumps());

    post_quench_ham = ising_ham(2);
    ψₜ = copy(ψ₀);
    envs = environments(ψₜ,post_quench_ham);

    echos = [echo(ψₜ,ψ₀)];
    times = collect(0:dt:finaltime);

    @showprogress for t = times[2:end]
        if t < 50*dt # if t is sufficiently small, we increase the bond dimension
            (ψₜ,envs) = changebonds(ψₜ,post_quench_ham,OptimalExpand(trscheme=truncdim(1)),envs)
        end
        (ψₜ,envs) = timestep(ψₜ,post_quench_ham,dt,Tdvp(),envs);
        push!(echos,echo(ψₜ,ψ₀))
    end

    return (times,echos)
end
```
![](infinite_timeev.png)
