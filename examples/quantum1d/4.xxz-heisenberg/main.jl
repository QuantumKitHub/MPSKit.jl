md"""
# The XXZ model

In this file we will give step by step instructions on how to analyze the spin 1/2 XXZ model.
The necessary packages to follow this tutorial are:
"""

using MPSKit, MPSKitModels, TensorKit, Plots

md"""
## Failure

First we should define the hamiltonian we want to work with.
Then we specify an initial guess, which we then further optimize.
Working directly in the thermodynamic limit, this is achieved as follows:
"""

H = heisenberg_XXX(; spin=1 // 2);

md"""
We then need an intial state, which we shall later optimize. In this example we work directly in the thermodynamic limit.
"""

random_data = TensorMap(rand, ComplexF64, ℂ^20 * ℂ^2, ℂ^20);
state = InfiniteMPS([random_data]);

md"""
The groundstate can then be found by calling `find_groundstate`.
"""

groundstate, cache, delta = find_groundstate(state, H, VUMPS());

md"""
As you can see, VUMPS struggles to converge.
On it's own, that is already quite curious.
Maybe we can do better using another algorithm, such as gradient descent.
"""

groundstate, cache, delta = find_groundstate(state, H, GradientGrassmann(; maxiter=20));

md"""
Convergence is quite slow and even fails after sufficiently many iterations.
To understand why, we can look at the transfer matrix spectrum.
"""

transferplot(groundstate, groundstate)

md"""
We can clearly see multiple eigenvalues close to the unit circle.
Our state is close to being non-injective, and represents the sum of multiple injective states.
This is numerically very problematic, but also indicates that we used an incorrect ansatz to approximate the groundstate.
We should retry with a larger unit cell.
"""

md"""
## Success

Let's initialize a different initial state, this time with a 2-site unit cell:
"""

A = TensorMap(rand, ComplexF64, ℂ^20 * ℂ^2, ℂ^20);
B = TensorMap(rand, ComplexF64, ℂ^20 * ℂ^2, ℂ^20);
state = InfiniteMPS([A, B]);

md"""
In MPSKit, we require that the periodicity of the hamiltonian equals that of the state it is applied to.
This is not a big obstacle, you can simply repeat the original hamiltonian.
Alternatively, the hamiltonian can be constructed directly on a two-site unitcell by making use of MPSKitModels.jl's `@mpoham`.
"""

## H2 = repeat(H, 2); -- copies the one-site version
H2 = heisenberg_XXX(ComplexF64, Trivial, InfiniteChain(2); spin=1 // 2)
groundstate, cache, delta = find_groundstate(state, H2,
                                             VUMPS(; maxiter=100, tol_galerkin=1e-12));

md"""
We get convergence, but it takes an enormous amount of iterations.
The reason behind this becomes more obvious at higher bond dimensions:
"""

groundstate, cache, delta = find_groundstate(state, H2,
                                             IDMRG2(; trscheme=truncdim(50), maxiter=20,
                                                    tol_galerkin=1e-12))
entanglementplot(groundstate)

md"""
We see that some eigenvalues clearly belong to a group, and are almost degenerate.
This implies 2 things:
- there is superfluous information, if those eigenvalues are the same anyway
- poor convergence if we cut off within such a subspace

It are precisely those problems that we can solve by using symmetries.
"""

md"""
## Symmetries

The XXZ Heisenberg hamiltonian is SU(2) symmetric and we can exploit this to greatly speed up the simulation.

It is cumbersome to construct symmetric hamiltonians, but luckily su(2) symmetric XXZ is already implemented:
"""

H2 = heisenberg_XXX(ComplexF64, SU2Irrep, InfiniteChain(2); spin=1 // 2);

md"""
Our initial state should also be SU(2) symmetric.
It now becomes apparent why we have to use a two-site periodic state.
The physical space carries a half-integer charge and the first tensor maps the first `virtual_space ⊗ the physical_space` to the second `virtual_space`.
Half-integer virtual charges will therefore map only to integer charges, and vice versa.
The staggering thus happens on the virtual level.

An alternative constructor for the initial state is
"""

P = Rep[SU₂](1 // 2 => 1)
V1 = Rep[SU₂](1 // 2 => 10, 3 // 2 => 5, 5 // 2 => 2)
V2 = Rep[SU₂](0 => 15, 1 => 10, 2 => 5)
state = InfiniteMPS([P, P], [V1, V2]);

md"""
Even though the bond dimension is higher than in the example without symmetry, convergence is reached much faster:
"""

println(dim(V1))
println(dim(V2))
groundstate, cache, delta = find_groundstate(state, H2,
                                             VUMPS(; maxiter=400, tol_galerkin=1e-12));
