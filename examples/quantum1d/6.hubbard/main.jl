md"""
# [Hubbard chain at half filling](@id hubbard)

The Hubbard model is a model of interacting fermions on a lattice, which is often used as a somewhat realistic model for electrons in a solid.
The Hamiltonian consists of two terms that describe competing forces of each electron:
a kinetic term that allows electrons to hop between neighboring sites, and a potential term reflecting on-site interactions between electrons.
Often, a third term is included which serves as a chemical potential to control the number of electrons in the system.

```math
H = -t \sum_{\langle i, j \rangle, \sigma} c^{\dagger}_{i,\sigma} c_{j,\sigma} + U \sum_i n_{i,\uparrow} n_{i,\downarrow} - \mu \sum_{i,\sigma} n_{i,\sigma}
```

At half-filling, the system exhibits particle-hole symmetry, which can be made explicit by rewriting the Hamiltonian slightly.
First, we fix the overall energy scale by setting `t = 1`, and then shift the total energy by adding a constant `U / 4`, as well as shifting the chemical potential to `N U / 2`.
This results in the following Hamiltonian:

```math
H = - \sum_{\langle i, j \rangle, \sigma} c^{\dagger}_{i,\sigma} c_{j,\sigma} + U / 4 \sum_i (1 - 2 n_{i,\uparrow}) (1 - 2 n_{i,\downarrow}) - \mu \sum_{i,\sigma} n_{i,\sigma}
```

Finally, setting `\mu = 0` and defining `u = U / 4` we obtain the Hubbard model at half-filling.

```math
H = - \sum_{\langle i, j \rangle, \sigma} c^{\dagger}_{i,\sigma} c_{j,\sigma} + u \sum_i (1 - 2 n_{i,\uparrow}) (1 - 2 n_{i,\downarrow})
```
"""

using TensorKit
using MPSKit
using MPSKitModels
using SpecialFunctions: besselj0, besselj1
using QuadGK: quadgk
using Plots

const t = 1.0
const mu = 0.0
const U = 3.0

md"""
For this case, the groundstate energy has an analytic solution, which can be used to benchmark the numerical results.
It follows from Eq. (6.82) in []().

```math
e(u) = - u - 4 \int_0^{\infty} \frac{d\omega}{\omega} \frac{J_0(\omega) J_1(\omega)}{1 + \exp(2u \omega)}
```

We can easily verify this by comparing the numerical results to the analytic solution.
"""

_integrandum(u, ω) = besselj0(ω) * besselj1(ω) / (1 + exp(2u * ω)) / ω
function hubbard_energy(u; rtol=1e-12)
    int, err = quadgk(Base.Fix1(_integrandum, u), 0, Inf; rtol)
    return -u - 4 * int
end

function compute_groundstate(psi, H; svalue=1e-3, expansion_factor=1 / 5, expansion_iter=20)
    # initial state
    psi, = find_groundstate(psi, H; tol=svalue * 10)

    # expansion steps
    for _ in 1:expansion_iter
        D = maximum(x -> dim(left_virtualspace(psi, x)), 1:length(psi))
        D′ = max(2, round(Int, D * expansion_factor))
        trscheme = truncbelow(svalue / 10) & truncdim(D′)
        psi′, = changebonds(psi, H, OptimalExpand(; trscheme))
        all(left_virtualspace.(Ref(psi), 1:length(psi)) .==
            left_virtualspace.(Ref(psi′), 1:length(psi))) && break
        psi, = find_groundstate(psi′, H, VUMPS(; tol=svalue / 5))
    end

    # convergence steps
    psi, = changebonds(psi, H, SvdCut(; trscheme=truncbelow(svalue)))
    psi, = find_groundstate(psi, H,
                            VUMPS(; tol=svalue) & GradientGrassmann(; tol=svalue / 100))

    return psi
end

H = hubbard_model(InfiniteChain(2); U, t, mu=U / 2)
psi = InfiniteMPS(H.data.pspaces, H.data.pspaces)
psi = compute_groundstate(psi, H)
E = real(expectation_value(psi, H)) / 2
@info """
Groundstate energy:
    * numerical: $E
    * analytic: $(hubbard_energy(U / 4) - U / 4)
"""

md"""
## Symmetries

The Hubbard model has a rich symmetry structure, which can be exploited to speed up simulations.
Apart from the fermionic parity, the model also has a $U(1)$ particle number symmetry, along with a $SU(2)$ spin symmetry.
Explicitly imposing these symmetries on the tensors can greatly reduce the computational cost of the simulation.

Naively imposing these symmetries however, is not compatible with our desire to work at half-filling.
By construction, imposing symmetries restricts the optimization procedure to a single symmetry sector, which is the trivial sector.
In order to work at half-filling, we need to effectively inject one particle per site.
In MPSKit, this is achieved by the `add_physical_charge` function, which shifts the physical spaces of the tensors to the desired charge sector.
"""

H_u1_su2 = hubbard_model(ComplexF64, U1Irrep, SU2Irrep, InfiniteChain(2); U, t, mu=U / 2);
charges = fill(FermionParity(1) ⊠ U1Irrep(1) ⊠ SU2Irrep(0), 2);
H_u1_su2 = MPSKit.add_physical_charge(H_u1_su2, dual.(charges));

pspaces = H_u1_su2.data.pspaces
vspaces = [oneunit(eltype(pspaces)), first(pspaces)]
psi = InfiniteMPS(pspaces, vspaces)
psi = compute_groundstate(psi, H_u1_su2; svalue=1e-3, expansion_factor=1 / 3,
                          expansion_iter=20)
E = real(expectation_value(psi, H_u1_su2)) / 2
@info """
Groundstate energy:
    * numerical: $E
    * analytic: $(hubbard_energy(U / 4) - U / 4)
"""

md"""
## Excitations

Because of the integrability, it is known that the Hubbard model has a rich excitation spectrum.
The elementary excitations are known as spinons and holons, which are domain walls in the spin and charge sectors, respectively.
The fact that the spin and charge sectors are separate is a phenomenon known as spin-charge separation.

The domain walls can be constructed by noticing that there are two equivalent groundstates, which differ by a translation over a single site.
In other words, the groundstates are ``\psi_{AB}` and ``\psi_{BA}``, where ``A`` and ``B`` are the two sites.
These excitations can then be constructed as follows:
"""

alg = QuasiparticleAnsatz(; tol=1e-3)
momenta = range(-π, π; length=12)
psi_AB = psi
envs_AB = environments(psi_AB, H_u1_su2);
psi_BA = circshift(psi, 1)
envs_BA = environments(psi_BA, H_u1_su2);

spinon_charge = FermionParity(0) ⊠ U1Irrep(0) ⊠ SU2Irrep(1 // 2)
E_spinon, ϕ_spinon = excitations(H_u1_su2, alg, momenta,
                                 psi_AB, envs_AB, psi_BA, envs_BA;
                                 sector=spinon_charge, num=1)

holon_charge = FermionParity(0) ⊠ U1Irrep(1) ⊠ SU2Irrep(0)
E_spinon, ϕ_spinon = excitations(H_u1_su2, alg, momenta,
                                 psi_AB, envs_AB, psi_BA, envs_BA;
                                 sector=holon_charge, num=1)

p_excitations = plot(; xaxis="momentum", yaxis="energy")
plot!(p_excitations, momenta, real(E_spinon); label="spinon")
plot!(p_excitations, momenta, real(E_holon); label="holon")
