using Markdown

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
using Interpolations
using Optim

const t = 1.0
const mu = 0.0
const U = 3.0

md"""
For this case, the ground state energy has an analytic solution, which can be used to benchmark the numerical results.
It follows from Eq. (6.82) in []().

```math
e(u) = - u - 4 \int_0^{\infty} \frac{d\omega}{\omega} \frac{J_0(\omega) J_1(\omega)}{1 + \exp(2u \omega)}
```

We can easily verify this by comparing the numerical results to the analytic solution.
"""

function hubbard_energy(u; rtol = 1.0e-12)
    integrandum(ω) = besselj0(ω) * besselj1(ω) / (1 + exp(2u * ω)) / ω
    int, err = quadgk(integrandum, 0, Inf; rtol = rtol)
    return -u - 4 * int
end

function compute_groundstate(
        psi, H;
        svalue = 1.0e-3,
        expansionfactor = (1 / 10),
        expansioniter = 20
    )
    verbosity = 2
    psi, = find_groundstate(psi, H; tol = svalue * 10, verbosity)
    for _ in 1:expansioniter
        D = maximum(x -> dim(left_virtualspace(psi, x)), 1:length(psi))
        D′ = max(5, round(Int, D * expansionfactor))
        trscheme = trunctol(; atol = svalue / 10) & truncrank(D′)
        psi′, = changebonds(psi, H, OptimalExpand(; trscheme = trscheme))
        all(
            left_virtualspace.(Ref(psi), 1:length(psi)) .==
                left_virtualspace.(Ref(psi′), 1:length(psi))
        ) && break
        psi, = find_groundstate(psi′, H, VUMPS(; tol = svalue / 5, maxiter = 10, verbosity))
    end

    ## convergence steps
    psi, = changebonds(psi, H, SvdCut(; trscheme = trunctol(; atol = svalue)))
    psi, = find_groundstate(
        psi, H,
        VUMPS(; tol = svalue / 100, verbosity, maxiter = 100) &
            GradientGrassmann(; tol = svalue / 1000)
    )

    return psi
end

H = hubbard_model(InfiniteChain(2); U, t, mu = U / 2)
Vspaces = fill(Vect[fℤ₂](0 => 10, 1 => 10), 2)
psi = InfiniteMPS(physicalspace(H), Vspaces)
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

H_u1_su2 = hubbard_model(ComplexF64, U1Irrep, SU2Irrep, InfiniteChain(2); U, t, mu = U / 2);
charges = fill(FermionParity(1) ⊠ U1Irrep(1) ⊠ SU2Irrep(0), 2);
H_u1_su2 = MPSKit.add_physical_charge(H_u1_su2, charges);

pspaces = physicalspace.(Ref(H_u1_su2), 1:2)
vspaces = [oneunit(eltype(pspaces)), first(pspaces)]
psi = InfiniteMPS(pspaces, vspaces)
psi = compute_groundstate(psi, H_u1_su2; expansionfactor = 1 / 3)
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
These excitations can be constructed as follows:
"""

alg = QuasiparticleAnsatz(; tol = 1.0e-3)
momenta = range(-π, π; length = 33)
psi_AB = psi
envs_AB = environments(psi_AB, H_u1_su2);
psi_BA = circshift(psi, 1)
envs_BA = environments(psi_BA, H_u1_su2);

spinon_charge = FermionParity(0) ⊠ U1Irrep(0) ⊠ SU2Irrep(1 // 2)
E_spinon, ϕ_spinon = excitations(
    H_u1_su2, alg, momenta, psi_AB, envs_AB, psi_BA, envs_BA;
    sector = spinon_charge, num = 1
);

holon_charge = FermionParity(1) ⊠ U1Irrep(-1) ⊠ SU2Irrep(0)
E_holon, ϕ_holon = excitations(
    H_u1_su2, alg, momenta, psi_AB, envs_AB, psi_BA, envs_BA;
    sector = holon_charge, num = 1
);

md"""
Again, we can compare the numerical results to the analytic solution.
Here, the formulae for the excitation energies are expressed in terms of dressed momenta:
"""

function spinon_momentum(Λ, u; rtol = 1.0e-12)
    integrandum(ω) = besselj0(ω) * sin(ω * Λ) / ω / cosh(ω * u)
    return π / 2 - quadgk(integrandum, 0, Inf; rtol = rtol)[1]
end
function spinon_energy(Λ, u; rtol = 1.0e-12)
    integrandum(ω) = besselj1(ω) * cos(ω * Λ) / ω / cosh(ω * u)
    return 2 * quadgk(integrandum, 0, Inf; rtol = rtol)[1]
end

function holon_momentum(k, u; rtol = 1.0e-12)
    integrandum(ω) = besselj0(ω) * sin(ω * sin(k)) / ω / (1 + exp(2u * abs(ω)))
    return π / 2 - k - 2 * quadgk(integrandum, 0, Inf; rtol = rtol)[1]
end
function holon_energy(k, u; rtol = 1.0e-12)
    integrandum(ω) = besselj1(ω) * cos(ω * sin(k)) * exp(-ω * u) / ω / cosh(ω * u)
    return 2 * cos(k) + 2u + 2 * quadgk(integrandum, 0, Inf; rtol = rtol)[1]
end

Λs = range(-10, 10; length = 51)
P_spinon_analytic = rem2pi.(spinon_momentum.(Λs, U / 4), RoundNearest)
E_spinon_analytic = spinon_energy.(Λs, U / 4)
I_spinon = sortperm(P_spinon_analytic)
P_spinon_analytic = P_spinon_analytic[I_spinon]
E_spinon_analytic = E_spinon_analytic[I_spinon]
P_spinon_analytic = [reverse(-P_spinon_analytic); P_spinon_analytic]
E_spinon_analytic = [reverse(E_spinon_analytic); E_spinon_analytic];

ks = range(0, 2π; length = 51)
P_holon_analytic = rem2pi.(holon_momentum.(ks, U / 4), RoundNearest)
E_holon_analytic = holon_energy.(ks, U / 4)
I_holon = sortperm(P_holon_analytic)
P_holon_analytic = P_holon_analytic[I_holon]
E_holon_analytic = E_holon_analytic[I_holon];

p = let p_excitations = plot(; xaxis = "momentum", yaxis = "energy")
    scatter!(p_excitations, momenta, real(E_spinon); label = "spinon")
    plot!(p_excitations, P_spinon_analytic, E_spinon_analytic; label = "spinon (analytic)")

    scatter!(p_excitations, momenta, real(E_holon); label = "holon")
    plot!(p_excitations, P_holon_analytic, E_holon_analytic; label = "holon (analytic)")

    p_excitations
end

md"""
The plot shows some discrepancies between the numerical and analytic results.
First and foremost, we must realize that in the thermodynamic limit, the momentum of a domain wall is actually not well-defined.
Concretely, only the difference in momentum between the two groundstates is well-defined, as we can always shift the momentum by multiplying one of the groundstates by a phase.
Here, we can fix this shift by realizing that our choice of shifting the groundstates by a single site, differs from the formula by a factor ``\pi/2``.
"""

momenta_shifted = rem2pi.(momenta .- π / 2, RoundNearest)
p = let p_excitations = plot(; xaxis = "momentum", yaxis = "energy", xlims = (-π, π))
    scatter!(p_excitations, momenta_shifted, real(E_spinon); label = "spinon")
    plot!(p_excitations, P_spinon_analytic, E_spinon_analytic; label = "spinon (analytic)")

    scatter!(p_excitations, momenta_shifted, real(E_holon); label = "holon")
    plot!(p_excitations, P_holon_analytic, E_holon_analytic; label = "holon (analytic)")

    p_excitations
end

md"""
The second discrepancy is that while the spinon dispersion is well-reproduced, the holon dispersion is not.
This is due to the fact that the excitation ansatz captures the lowest-energy excitation, and not the elementary single-particle excitation.
To make this explicit, we can consider the scattering states comprising of a holon and two spinons.
If these are truly scattering states, the energy of the scattering state should be the sum of the energies of the individual excitations, and the momentum is the sum of the momenta.
Thus, we can find the lowest-energy scattering states by minimizing the energy over the combination of momenta for the constituent elementary excitations.
"""

holon_dispersion_itp = linear_interpolation(
    P_holon_analytic, E_holon_analytic;
    extrapolation_bc = Line()
)
spinon_dispersion_itp = linear_interpolation(
    P_spinon_analytic, E_spinon_analytic;
    extrapolation_bc = Line()
)
function scattering_energy(p1, p2, p3)
    p1, p2, p3 = rem2pi.((p1, p2, p3), RoundNearest)
    return holon_dispersion_itp(p1) + spinon_dispersion_itp(p2) + spinon_dispersion_itp(p3)
end;

E_scattering_min = map(momenta_shifted) do p
    e = Inf
    for i in 1:10 # repeat for stability
        res = optimize((rand(2) .* (2π) .- π)) do (p₁, p₂)
            p₃ = p - p₁ - p₂
            return scattering_energy(p₁, p₂, p₃)
        end

        e = min(Optim.minimum(res), e)
    end
    return e
end
E_scattering_max = map(momenta_shifted) do p
    e = -Inf
    for i in 1:10 # repeat for stability
        res = optimize((rand(Float64, 2) .* (2π) .- π)) do (p₁, p₂)
            p₃ = p - p₁ - p₂
            return -scattering_energy(p₁, p₂, p₃)
        end

        e = max(-Optim.minimum(res), e)
    end
    return e
end;

p = let p_excitations = plot(;
        xaxis = "momentum", yaxis = "energy", xlims = (-π, π), ylims = (-0.1, 5)
    )
    scatter!(p_excitations, momenta_shifted, real(E_spinon); label = "spinon")
    plot!(p_excitations, P_spinon_analytic, E_spinon_analytic; label = "spinon (analytic)")

    scatter!(p_excitations, momenta_shifted, real(E_holon); label = "holon")
    plot!(p_excitations, P_holon_analytic, E_holon_analytic; label = "holon (analytic)")

    I = sortperm(momenta_shifted)
    plot!(
        p_excitations, momenta_shifted[I], E_scattering_min[I]; label = "scattering states",
        fillrange = E_scattering_max[I], fillalpha = 0.3, fillstyle = :x
    )

    p_excitations
end
