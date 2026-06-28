using Markdown
using MPSKit, TensorKit
using TensorKitTensors.SpinOperators: spin_space, S_z, S_exchange
using MPSKit: BlockTensorKit.eachspace
using KrylovKit: eigsolve, exponentiate
using TensorOperations
using Plots

#src # for reproducibility:
#src using Random
#src Random.seed!(123)

md"""
# Dynamical spin-spin correlator in the Heisenberg chain

This example computes the real-time dynamical spin correlator of the spin-½ antiferromagnetic Heisenberg chain,

```math
C_{ij}(t) = \langle \psi_0 | \vec S_j(t) \cdot \vec S_i(0) | \psi_0 \rangle
          = \sum_{a} \langle \psi_0 | S_j^a(t)\, S_i^a(0) | \psi_0 \rangle ,
```

with ``\vec S_j(t) = e^{iHt}\vec S_j e^{-iHt}`` and ``H = \sum_i \vec S_i \cdot \vec S_{i+1}``.
Its space-time Fourier transform is the dynamical structure factor ``S(q,ω)``, whose spectral weight in this model forms the famous *spinon continuum*.
For an infinite chain the continuum is bounded below and above by the des Cloizeaux–Pearson edges,

```math
ω_L(q) = \tfrac{π}{2}\,|\sin q|, \qquad ω_U(q) = π\,\left|\sin\tfrac{q}{2}\right| ,
```

which we will overlay on the computed structure factor as a sanity check.

The exchange ``\vec S_i \cdot \vec S_j`` is not a tensor product of single-site operators, so we cannot simply apply it as one local gate per site.
The trick used throughout this example is to *factorize* the exchange into two halves joined by an **auxiliary index**, promote that auxiliary index to a temporary physical site, evolve while acting trivially on it, and finally contract it away when closing the correlator.

We build up the method in three stages, each reproducing the same numbers: exact diagonalization (the reference), MPS without symmetry, and MPS with `SU(2)` symmetry.
"""

T = ComplexF64

md"""
The Hamiltonian is generic in the symmetry sector, so the *same* constructor serves all three runs.
"""

function heisenberg_hamiltonian(L, ::Type{I} = Trivial; J::Real = 1.0) where {I <: Sector}
    @assert iseven(L) "Spin 1/2 requires an even number of sites ($L)"
    spin = 1 // 2
    SS = scale!!(S_exchange(T, I; spin), J)
    lattice = fill(spin_space(I; spin), L)
    return FiniteMPOHamiltonian(lattice, (i, i + 1) => SS for i in 1:(L - 1))
end

md"""
## Method 1 — Exact diagonalization baseline

On a small chain we can diagonalize the Hamiltonian densely and obtain an unambiguous reference.
We keep this system tiny (`L_ed = 6`): the dense Hamiltonian construction is the compile-time bottleneck, and six sites already capture the relevant physics for a benchmark.
"""

L_ed = 6
H_ed = real(convert(TensorMap, heisenberg_hamiltonian(L_ed)))
Es, ψs, = eigsolve(x -> H_ed * x, rand(T, (ℂ^2)^L_ed), 1, :SR; ishermitian = true)
ψ₀_ed = first(ψs)
E₀_ed = real(Es[1])

md"""
We evaluate ``C^{zz}_{ij}(t) = \langle ψ_0 | S_j^z e^{-i(H-E_0)t} S_i^z | ψ_0 \rangle`` directly by applying ``S^z`` to the ground state, evolving the result with `KrylovKit`'s exponentiator, and contracting against ``S^z|ψ_0\rangle`` at every site.

We shift the Hamiltonian by the ground-state energy ``E_0``, because ``\langle ψ_0|e^{iHt} = e^{iE_0 t}\langle ψ_0|`` contributes only a trivial global phase that we remove by replacing ``H \to H - E_0``.
By `SU(2)` symmetry the full vector correlator is exactly three times the longitudinal one, ``C_{ij} = \langle \vec S_j \cdot \vec S_i\rangle = 3\, C^{zz}_{ij}``, which is what we will compare the MPS results to.
"""

function ed_czz(H_ed, ψ₀, E₀, L, times)
    δt = step(times)
    C = zeros(ComplexF64, L, L, length(times))
    Z = S_z(Float64)
    for i in 1:L
        Sψ = ncon([Z, ψ₀], [[-i, 1], replace(-(1:L), -i => 1)])
        for (it, t) in enumerate(times)
            it > 1 && ((Sψ,) = exponentiate(x -> H_ed * x - E₀ * x, -δt * im, Sψ; ishermitian = true))
            for j in 1:L
                C[i, j, it] = ncon(
                    [ψ₀, Z, Sψ],
                    [collect(2:(L + 1)), [j + 1, 1], replace(2:(L + 1), j + 1 => 1)],
                    [true, false, false],
                )
            end
        end
    end
    return C
end

δt = 0.05
maxtime = 6.0
times = 0:δt:maxtime

Cᶻᶻ_ed = ed_czz(H_ed, ψ₀_ed, E₀_ed, L_ed, times)

md"""
A quick physical check: the equal-time on-site value should be ``\langle \vec S_i \cdot \vec S_i \rangle = S(S+1) = 3/4``, and nearest neighbours should be antiferromagnetically correlated (negative).
"""

@assert isapprox(3 * Cᶻᶻ_ed[1, 1, 1], 0.75; atol = 1.0e-8)
@assert real(3 * Cᶻᶻ_ed[1, 2, 1]) < 0

md"""
## Method 2 — MPS without symmetry

We now reproduce the reference with matrix product states.
The ground state is obtained with two-site DMRG, starting from a Néel product state.
"""

function neel_state(L, ::Type{Trivial} = Trivial)
    P = spin_space(Trivial; spin = 1 // 2)
    tensors = map(1:L) do i
        A = zeros(T, oneunit(P) ⊗ P ← oneunit(P))
        A[1, iseven(i) ? 1 : 2, 1] = 1
        return A
    end
    return FiniteMPS(tensors)
end

md"""
### Factorizing the exchange

We split the two-site exchange ``\vec S_i \cdot \vec S_j`` into a left and a right factor connected by an auxiliary index `a`.
A singular value decomposition does exactly this: it produces `S_left[j₁ a; i₁]` and `S_right[j₂; a i₂]` whose contraction over `a` rebuilds the exchange.
For the trivial-symmetry case the auxiliary space is three-dimensional — it simply enumerates the ``a \in \{x,y,z\}`` components.
"""

function factorize(SS)
    @tensor SS_perm[i1 j1; i2 j2] := SS[j1 j2; i1 i2]
    U, S, Vᴴ = svd_trunc!(SS_perm; trunc = trunctol(; atol = eps(real(scalartype(SS)))^(3 / 4)))
    sqrtS = sqrt(S)
    @tensor begin
        S_left[j1 a; i1] := U[i1 j1; b] * sqrtS[b; a]
        S_right[j2; a i2] := sqrtS[a; b] * Vᴴ[b; i2 j2]
    end
    return S_left, S_right
end

md"""
### Promoting the auxiliary index to a physical site

Applying `S_left` at the source site leaves a dangling auxiliary leg.
We absorb it as a *new physical site* inserted right after the source: the chain temporarily grows from ``N`` to ``N+1`` sites, the extra site carrying the auxiliary space.
"""

function apply_split(state, O, i)
    state = copy(state)
    if numout(O) == 2       # left factor: opens an auxiliary leg `p2`
        @plansor A2[l p1; r p2] := state.AC[i][l p1'; r] * O[p1 p2; p1']
    elseif numin(O) == 2    # right factor (used as a closing operator)
        @plansor A2[l p1; r p2] := state.AC[i][l p1'; r] * O[p2; p1 p1']
    end
    Al, Ar = left_orth!(A2)
    Ar = MPSKit._transpose_front(Ar)
    As = collect(eltype(state), vcat(state.AL[1:(i - 1)], [Al, Ar], state.AR[(i + 1):end]))
    return FiniteMPS(As; normalize = false)
end

md"""
### Lifting the Hamiltonian over the auxiliary site

Time evolution acts through a `FiniteMPOHamiltonian`, so the inserted auxiliary site must itself be a valid Jordan-form Hamiltonian tensor that acts trivially.
We thread *every* virtual index of the MPO through the new site with a `BraidingTensor` on the diagonal: this passes both the boundary identity strings and the half-finished nearest-neighbour coupling straight across, so the physical sites on either side remain coupled as if the auxiliary site were not there.
"""

function insert_id_op(H, P, i)
    H′ = copy(H)
    W = typeof(H[1])(undef, left_virtualspace(H, i) ⊗ P ← P ⊗ left_virtualspace(H, i))
    for row in 1:size(W, 1)
        W[row, 1, 1, row] = BraidingTensor{scalartype(H), spacetype(H), storagetype(H)}(
            eachspace(W)[CartesianIndex(row, 1, 1, row)]
        )
    end
    insert!(parent(H′), i, W)
    return H′
end

md"""
### Closing the correlator

After evolving, we contract the right factor `S_right` at the sink site against the static ground state, threading the auxiliary leg from the auxiliary site to the sink (in whichever order they appear along the chain).
This evaluates ``\langle ψ_0 | S_j^{\text{right}}\, e^{-i(H-E_0)t}\, S_i^{\text{left}} | ψ_0\rangle`` for a fixed source `i₀` and every sink `sink`.
"""

function excited_overlap(bra, S_right, sink, ket, i₀)
    N = length(bra)
    @assert length(ket) == N + 1
    aux_pos = i₀ + 1
    sink_pos = sink ≤ i₀ ? sink : sink + 1
    brasite(p) = p ≤ i₀ ? p : p - 1

    @tensor L[b; t] := conj(bra.C[0][m; b]) * ket.C[0][m; t]
    m_open = false
    for p in 1:(N + 1)
        if p == sink_pos
            bs = brasite(p)
            if m_open    # auxiliary leg already threaded → close it through S_right
                @tensor Ln[b; t] := L[b' m; t'] * ket.AR[p][t' q; t] * S_right[q'; m q] *
                    conj(bra.AR[bs][b' q'; b])
                m_open = false
            else         # open the auxiliary leg via S_right
                @tensor Ln[b m; t] := L[b'; t'] * ket.AR[p][t' q; t] * S_right[q'; m q] *
                    conj(bra.AR[bs][b' q'; b])
                m_open = true
            end
            L = Ln
        elseif p == aux_pos
            if m_open    # close the auxiliary leg against the auxiliary site (no bra here)
                @tensor Ln[b; t] := L[b m; t'] * ket.AR[p][t' m; t]
                m_open = false
            else         # open the auxiliary leg from the auxiliary site
                @tensor Ln[b m; t] := L[b; t'] * ket.AR[p][t' m; t]
                m_open = true
            end
            L = Ln
        else
            bs = brasite(p)
            if m_open
                @tensor Ln[b m; t] := L[b' m; t'] * ket.AR[p][t' q; t] * conj(bra.AR[bs][b' q; b])
            else
                @tensor Ln[b; t] := L[b'; t'] * ket.AR[p][t' q; t] * conj(bra.AR[bs][b' q; b])
            end
            L = Ln
        end
    end
    return @tensor L[a; a]
end

md"""
### Putting it together

The full procedure for a fixed source `i₀`: apply the left factor (inserting the auxiliary site), evolve once with the lifted, energy-shifted Hamiltonian using two-site TDVP (which lets the bond dimension grow), and close at every sink at each stored time.
A single time evolution thus yields ``C_{i_0, j}(t)`` for all sinks ``j``.
"""

function correlator_from_source(gs, S_left, S_right, H, E₀, i₀, times; χ = 64)
    φ = apply_split(gs, S_left, i₀)
    H′ = insert_id_op(H, space(S_left, 2), i₀ + 1)
    H_shift = H′ - fill(E₀ / length(H′), length(H′))
    N = length(gs)
    C = zeros(ComplexF64, N, length(times))
    envs = environments(φ, H_shift)
    δt = step(times)
    for (it, t) in enumerate(times)
        it > 1 && ((φ, envs) = timestep(φ, H_shift, t, δt, TDVP2(; trscheme = truncrank(χ)), envs))
        for sink in 1:N
            C[sink, it] = excited_overlap(gs, S_right, sink, φ, i₀)
        end
    end
    return C
end

md"""
We validate against the exact-diagonalization reference on the same ``L = 6`` chain.
With the source in the middle of the chain, the full vector correlator should equal ``3\, C^{zz}_{i_0 j}(t)`` from method 1.
"""

H_triv = heisenberg_hamiltonian(L_ed)
gs_triv, = find_groundstate(neel_state(L_ed), H_triv, DMRG2(; maxiter = 30, trscheme = truncrank(64)))
E₀_triv = real(expectation_value(gs_triv, H_triv))

SS_triv = S_exchange(T)
S_left_triv, S_right_triv = factorize(SS_triv)
@tensor SS_check[j1 j2; i1 i2] := S_left_triv[j1 a; i1] * S_right_triv[j2; a i2]
@assert SS_check ≈ SS_triv

i₀ = L_ed ÷ 2
C_triv = correlator_from_source(gs_triv, S_left_triv, S_right_triv, H_triv, E₀_triv, i₀, times)

err_triv = maximum(abs, C_triv[j, it] - 3 * Cᶻᶻ_ed[i₀, j, it] for j in 1:L_ed, it in 1:length(times))
@info "Trivial MPS vs 3·ED Cᶻᶻ" err_triv
@assert err_triv < 1.0e-8

md"""
## Method 3 — MPS with `SU(2)` symmetry

Nothing about the construction is specific to the absence of symmetry: `apply_split`, `insert_id_op` and `excited_overlap` are all written with `@tensor`/`BraidingTensor` and work for any symmetry.
We only need an `SU(2)`-symmetric initial state.

A natural choice is a **dimer (valence-bond) state**, pairing neighbouring sites into total-spin singlets.
Its virtual spaces alternate between a singlet (between dimers) and a spin-½ bond (within a dimer), and it is an exact total-singlet starting point for DMRG.
"""

function dimer_state(L)
    @assert iseven(L)
    P = spin_space(SU2Irrep; spin = 1 // 2)
    Vhalf = SU2Space(1 // 2 => 1)
    Vsinglet = oneunit(P)
    tensors = map(1:L) do k
        isodd(k) ? isometry(T, Vsinglet ⊗ P, Vhalf) : isometry(T, Vhalf ⊗ P, Vsinglet)
    end
    return FiniteMPS(tensors)
end

H_su2 = heisenberg_hamiltonian(L_ed, SU2Irrep)
gs_su2, = find_groundstate(dimer_state(L_ed), H_su2, DMRG2(; maxiter = 30, trscheme = truncrank(64)))
E₀_su2 = real(expectation_value(gs_su2, H_su2))

md"""
Factorizing the `SU(2)` exchange gives an auxiliary leg in the spin-1 channel (`Rep[SU₂](1)`): the exchange transforms as a rank-1 spherical tensor.
The exact same orchestration then reproduces the trivial-symmetry result to machine precision.
"""

SS_su2 = S_exchange(T, SU2Irrep)
S_left_su2, S_right_su2 = factorize(SS_su2)

C_su2 = correlator_from_source(gs_su2, S_left_su2, S_right_su2, H_su2, E₀_su2, i₀, times)

err_su2 = maximum(abs, C_su2[j, it] - C_triv[j, it] for j in 1:L_ed, it in 1:length(times))
@info "SU(2) MPS vs trivial MPS" err_su2
@assert err_su2 < 1.0e-8

md"""
## The dynamical structure factor

To obtain ``S(q,ω)`` we Fourier transform the single-source correlator ``C_{i_0 j}(t)`` in both the sink position (relative to the source) and time.
We apply a Hann window in time to suppress the spectral leakage that the finite time window would otherwise produce.
"""

function structure_factor(C_source, i₀, times; window::Bool = true)
    N, Nt = size(C_source)
    δt = step(times)
    qs = 2π .* (0:(N - 1)) ./ N
    w = window ? (0.5 .* (1 .- cos.(2π .* (0:(Nt - 1)) ./ (Nt - 1)))) : ones(Nt)

    Sqω = zeros(ComplexF64, N, Nt)
    ωs = 2π .* (0:(Nt - 1)) ./ (Nt * δt)
    for (iq, q) in enumerate(qs)
        ## spatial transform relative to the source site
        Cq = [sum(cis(-q * (j - i₀)) * C_source[j, it] for j in 1:N) for it in 1:Nt]
        Cq .*= w
        for (iω, ω) in enumerate(ωs)
            ## temporal transform; e^{+iωt} places the excitations at positive ω
            Sqω[iq, iω] = δt * sum(cis(ω * times[it]) * Cq[it] for it in 1:Nt)
        end
    end
    return qs, ωs, Sqω
end

function plot_structure_factor(C_source, i₀, times; ωmax = 4.0, clip = 1.0, kwargs...)
    qs, ωs, Sqω = structure_factor(C_source, i₀, times; kwargs...)
    nω = findlast(≤(ωmax), ωs)
    S = abs.(permutedims(Sqω[:, 1:nω]))

    ## the quasi-elastic peak at q = π is much brighter than the continuum, so we
    ## optionally clip the colour scale (at the `clip` quantile) to make the continuum visible
    Ssorted = sort(vec(S))
    cmax = Ssorted[clamp(ceil(Int, clip * length(Ssorted)), 1, length(Ssorted))]

    plt = heatmap(
        qs ./ π, ωs[1:nω], S;
        clims = (0, cmax),
        xlabel = "q / π", ylabel = "ω", title = "Dynamical structure factor |S(q, ω)|",
        colorbar_title = "|S|",
    )
    ## des Cloizeaux–Pearson spinon continuum edges
    qfine = range(0, 2π; length = 200)
    plot!(plt, qfine ./ π, (π / 2) .* abs.(sin.(qfine)); color = :white, lw = 2, ls = :dash, label = "ω_L")
    plot!(plt, qfine ./ π, π .* abs.(sin.(qfine ./ 2)); color = :cyan, lw = 2, ls = :dash, label = "ω_U")
    return plt
end

md"""
A six-site chain is too small to resolve the continuum, but it already shows the spectral weight sitting between the two edges.
"""

plt_ed = plot_structure_factor(Cᶻᶻ_ed[i₀, :, :], i₀, times)

md"""
### Scaling up

To actually resolve the spinon continuum we run a larger `SU(2)` simulation.
We place the source at the centre of the chain and evolve to a fairly long final time to get a fine frequency resolution (``Δω ≈ 2π / t_{\max}``), while keeping it short enough that the correlations do not reach the open boundaries — the spinon velocity is ``v = π/2``, so the light-cone front stays well inside the chain.
"""

L = 48
χ = 48
times_big = 0:0.1:12.0

H_big = heisenberg_hamiltonian(L, SU2Irrep)
gs_big, = find_groundstate(dimer_state(L), H_big, DMRG2(; maxiter = 30, trscheme = truncrank(χ)))
E₀_big = real(expectation_value(gs_big, H_big))

S_left_big, S_right_big = factorize(S_exchange(T, SU2Irrep))
C_big = correlator_from_source(gs_big, S_left_big, S_right_big, H_big, E₀_big, L ÷ 2, times_big; χ = χ)

plt_big = plot_structure_factor(C_big, L ÷ 2, times_big; clip = 0.97)

md"""
The spectral weight clearly fills the des Cloizeaux–Pearson continuum, gapless at ``q = 0`` and ``q = π``, exactly as expected for the spin-½ Heisenberg chain — and it was obtained by the same auxiliary-space construction that we validated against exact diagonalization.
"""
