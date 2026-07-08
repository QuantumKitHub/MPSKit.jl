println("
-----------------------------
|   BUG time-stepping tests |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using LinearAlgebra: dot, norm
using Random

@testset "BUG time evolution" verbose = true begin
    dt = 0.1
    L = 10

    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
    ψ = FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4)
    ψ₀, = find_groundstate(ψ, H; verbosity = 0)
    E₀ = expectation_value(ψ₀, H)

    # 1. energy conservation + eigenstate phase
    @testset "energy conservation" begin
        ψ1, envs = timestep(ψ₀, H, 0.0, dt, BUG())
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
    end

    # 2. agreement with TDVP over a few real-time steps of a random MPS
    @testset "agreement with TDVP" begin
        Random.seed!(1234)
        ψr = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        δt = 0.01
        ψ_bug, ψ_tdvp = ψr, ψr
        for k in 0:4
            ψ_bug, = timestep(ψ_bug, H, k * δt, δt, BUG())
            ψ_tdvp, = timestep(ψ_tdvp, H, k * δt, δt, TDVP())
        end
        @test expectation_value(ψ_bug, H) ≈ expectation_value(ψ_tdvp, H) atol = 1.0e-3
        @test abs(dot(ψ_bug, ψ_tdvp)) ≈ 1 atol = 1.0e-3
    end

    # 3. second-order convergence on a small full-rank system (isolates the temporal order)
    @testset "second-order convergence" begin
        Random.seed!(2)
        Lc = 4
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψ_full = FiniteMPS(rand, ComplexF64, Lc, ℙ^2, ℙ^4)   # full-rank: 1,2,4,2,1

        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψ_full)
        ψvec /= norm(ψvec)

        T = 0.5
        dts = [0.1, 0.05, 0.025]
        errs = map(dts) do δt
            n = round(Int, T / δt)
            ref = exp(-im * Hmat * (n * δt)) * ψvec
            ψ = copy(ψ_full)
            envs = environments(ψ, Hc, ψ)
            for k in 0:(n - 1)
                timestep!(ψ, Hc, k * δt, δt, BUG(), envs)
            end
            ψout = convert(TensorMap, ψ)
            ψout /= norm(ψout)
            return 1 - abs(dot(ψout, ref))
        end

        slopes = [
            (log(errs[i + 1]) - log(errs[i])) / (log(dts[i + 1]) - log(dts[i]))
                for i in 1:(length(dts) - 1)
        ]
        @info "BUG convergence" errs slopes
        for s in slopes
            @test s ≈ 2 atol = 0.3
        end
    end

    # 4. imaginary-time evolution lowers the energy toward the ground state (and, having no
    #    backward substep, stays norm-preserving/stable)
    @testset "imaginary-time lowers energy" begin
        Random.seed!(5)
        ψi = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        E_start = real(expectation_value(ψi, H))
        E_prev = E_start
        for _ in 1:20
            ψi, = timestep(ψi, H, 0.0, 0.1, BUG(); imaginary_evolution = true)
            E_now = real(expectation_value(ψi, H))
            @test E_now ≤ E_prev + 1.0e-6   # monotone (non-increasing) energy
            E_prev = E_now
        end
        @test E_prev < E_start - 1.0        # substantial lowering toward the ground state
        @test norm(ψi) ≈ 1 atol = 1.0e-6    # imaginary-time BUG renormalizes each step
    end

    # 5. LazySum / MultipliedOperator smoke tests
    @testset "LazySum" begin
        Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])
        ψl, envs = timestep(ψ₀, Hlazy, 0.0, dt, BUG())
        E = expectation_value(ψl, Hlazy, envs)
        @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1.0e-2
    end

    @testset "TimeDependent LazySum" begin
        Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)
        ψa, envsa = timestep(ψ₀, Ht(1.0), 0.0, dt, BUG())
        Ea = expectation_value(ψa, Ht(1.0), envsa)

        ψt, envst = timestep(ψ₀, Ht, 1.0, dt, BUG())
        Et = expectation_value(ψt, Ht(1.0), envst)
        @test Ea ≈ Et atol = 1.0e-8
    end
end

# Rank-adaptive BUG (Stage 2): a truncating `trscheme` enables basis augmentation + truncation, so
# the bond dimension grows and shrinks to track the entanglement of the evolving state. Trivial
# tensors only (symmetry stress is Chunk 2.3). The default `notrunc()` regression is covered by the
# fixed-rank testsets above (they must all still pass unchanged).
@testset "BUG rank-adaptive" verbose = true begin
    # 1. bond growth: a tight tolerance grows a low-bond-dim state, a looser tolerance keeps it
    #    smaller.
    @testset "bond growth" begin
        Random.seed!(101)
        L = 6
        H = force_planar(transverse_field_ising(ComplexF64, Trivial; L))
        ψ₀ = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^2))   # low bond dim (2)
        normalize!(ψ₀)
        Dstart = maximum(dim(left_virtualspace(ψ₀, k)) for k in 1:L)

        ψtight = ψ₀
        for k in 0:2
            ψtight, = timestep(ψtight, H, k * 0.05, 0.05, BUG(; trscheme = truncerror(; atol = 1.0e-10)))
        end
        Dtight = maximum(dim(left_virtualspace(ψtight, k)) for k in 1:L)

        ψloose = ψ₀
        for k in 0:2
            ψloose, = timestep(ψloose, H, k * 0.05, 0.05, BUG(; trscheme = truncerror(; atol = 1.0e-2)))
        end
        Dloose = maximum(dim(left_virtualspace(ψloose, k)) for k in 1:L)

        @info "BUG rank-adaptive bond growth" Dstart Dloose Dtight
        @test Dtight > Dstart          # rank-adaptivity grows the bond
        @test Dloose < Dtight          # a looser tolerance keeps a smaller bond
    end

    # 2. THE HARD GATE: overlap-error vs the dense exp(-iHT) reference decreases (monotonically, up
    #    to the plateau at the fixed-dt floor) as the truncation tolerance ϑ shrinks. This proves the
    #    augmentation actually captures the true dynamics.
    @testset "accuracy improves as ϑ decreases" begin
        Random.seed!(202)
        L = 6
        H = force_planar(transverse_field_ising(ComplexF64, Trivial; L))
        ψ₀ = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^2))   # low-rank start the dynamics grows
        normalize!(ψ₀)

        Hmat = convert(TensorMap, H)
        ψvec = convert(TensorMap, ψ₀)
        ψvec /= norm(ψvec)

        T = 0.2
        dt = 0.05
        n = round(Int, T / dt)
        ref = exp(-im * Hmat * (n * dt)) * ψvec

        ϑs = [1.0e-2, 1.0e-4, 1.0e-6]
        errs = map(ϑs) do ϑ
            alg = BUG(; trscheme = truncerror(; atol = ϑ))
            ψ = copy(ψ₀)
            envs = environments(ψ, H, ψ)
            for k in 0:(n - 1)
                timestep!(ψ, H, k * dt, dt, alg, envs)
            end
            ψout = convert(TensorMap, ψ)
            ψout /= norm(ψout)
            return 1 - abs(dot(ψout, ref))
        end

        @info "BUG rank-adaptive accuracy vs ϑ" ϑs errs
        for i in 1:(length(ϑs) - 1)
            @test errs[i + 1] ≤ 1.5 * errs[i]   # monotone within plateau noise near the dt-floor
        end
        @test errs[end] < errs[1] / 10          # clear net improvement toward the dt-floor
    end

    # 3. CBE-style comparison: from a low-rank state, rank-adaptive BUG tracks a bond-adaptive TDVP2
    #    reference better than fixed-rank `BUG()` does (mirrors the CBE-TDVP test).
    @testset "tracks TDVP2 better than fixed-rank BUG" begin
        Random.seed!(303)
        L = 8
        H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
        Dstart, Dcap, dt = 2, 16, 0.05
        ψ₀ = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^Dstart))

        ref, adaptive, fixed = ψ₀, ψ₀, ψ₀
        for _ in 1:6
            ref, = timestep(ref, H, 0.0, dt, TDVP2(; trscheme = truncrank(Dcap)))
            adaptive, = timestep(adaptive, H, 0.0, dt, BUG(; trscheme = truncerror(; atol = 1.0e-8)))
            fixed, = timestep(fixed, H, 0.0, dt, BUG())
        end

        @test dim(left_virtualspace(adaptive, L ÷ 2)) > Dstart   # adaptive grew the bond
        @test dim(left_virtualspace(fixed, L ÷ 2)) == Dstart     # fixed-rank stuck at Dstart
        @test abs(dot(ref, adaptive)) > abs(dot(ref, fixed))     # and tracks the reference better
    end

    # 4. imaginary-time ground-state search: from a low bond dim, rank-adaptive imaginary-time BUG
    #    grows the bond and lowers the energy toward the true ground state.
    @testset "imaginary-time grows bond and lowers energy" begin
        Random.seed!(404)
        L = 8
        H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
        ψgs, = find_groundstate(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^16), H; verbosity = 0)
        Egs = real(expectation_value(ψgs, H))

        ψ = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^2))   # low bond dim
        Dstart = maximum(dim(left_virtualspace(ψ, k)) for k in 1:L)
        E_start = real(expectation_value(ψ, H))
        for _ in 1:30
            ψ, = timestep(ψ, H, 0.0, 0.1, BUG(; trscheme = truncerror(; atol = 1.0e-8)); imaginary_evolution = true)
        end
        Dend = maximum(dim(left_virtualspace(ψ, k)) for k in 1:L)
        E_end = real(expectation_value(ψ, H))

        @info "BUG rank-adaptive imaginary-time" Dstart Dend E_start E_end Egs
        @test Dend > Dstart              # the bond grew as entanglement built up
        @test E_end < E_start - 1.0      # substantial lowering
        @test E_end ≈ Egs atol = 0.6     # toward the true ground state (loose)
        @test norm(ψ) ≈ 1 atol = 1.0e-6  # imaginary-time renormalizes each step
    end

    # 5. real-time energy conservation with truncation (loose atol) and norm behaviour: for a small
    #    tolerance the truncation is negligible, so the norm stays ≈ 1.
    @testset "real-time energy conservation and norm" begin
        Random.seed!(505)
        L = 6
        H = force_planar(transverse_field_ising(ComplexF64, Trivial; L))
        ψ₀ = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        normalize!(ψ₀)
        E₀ = real(expectation_value(ψ₀, H))

        ψ = ψ₀
        for k in 0:4
            ψ, = timestep(ψ, H, k * 0.05, 0.05, BUG(; trscheme = truncerror(; atol = 1.0e-8)))
        end
        @test real(expectation_value(ψ, H)) ≈ E₀ atol = 1.0e-2   # energy conserved (loose)
        @test norm(ψ) ≈ 1 atol = 1.0e-6                          # tiny ϑ ⇒ norm preserved
    end
end

# Charge-sector (symmetric-tensor) coverage for the fixed-rank BUG. These use *genuine*
# symmetric tensors (no `force_planar`), exercising the graded-bond paths flagged in the design
# doc's hsector risk register (H1/H6/H7): the transport-tensor seed `isomorphism(V ← V)`, the
# `@plansor` (co)domain/dual conventions in `_bug_transport_*`, and the adjoints carrying sector
# duals. A fixed-rank step must preserve the total charge and the graded structure of every bond.
@testset "BUG symmetric tensors" verbose = true begin
    dt = 0.1
    L = 6

    # 1. U(1)-symmetric Heisenberg, both in the natural total-Sz = 0 sector and in a fixed nonzero
    #    total-charge (Sz = 1) sector: energy conservation + eigenstate phase + sector preservation.
    @testset "U(1) Heisenberg (total Sz = $label)" for (label, right) in
        (("0", U1Space(0 => 1)), ("1", U1Space(1 => 1)))
        Random.seed!(2718)
        H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψ = FiniteMPS(physicalspace(H), maxV[2:(end - 1)]; right)
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)

        Vl₀ = left_virtualspace.(Ref(ψ₀), 1:L)
        Vr₀ = right_virtualspace.(Ref(ψ₀), 1:L)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, BUG())
        E1 = expectation_value(ψ1, H, envs)

        @test E₀ ≈ E1 atol = 1.0e-2
        @test imag(E1) ≈ 0 atol = 1.0e-8
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        # the fixed-rank step preserves the graded structure (sector content) of every bond
        @test left_virtualspace.(Ref(ψ1), 1:L) == Vl₀
        @test right_virtualspace.(Ref(ψ1), 1:L) == Vr₀
    end

    # 2. A second symmetry group. Z2 (transverse-field Ising) and SU2 (Heisenberg) both stress the
    #    graded transport tensor; same assertions (energy conservation + eigenstate phase).
    @testset "Z2 transverse-field Ising" begin
        Random.seed!(161803)
        H = transverse_field_ising(ComplexF64, Z2Irrep; g = 1.0, L)
        ψ = FiniteMPS(physicalspace(H), Z2Space(0 => 4, 1 => 4))
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)
        Vl₀ = left_virtualspace.(Ref(ψ₀), 1:L)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, BUG())
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test left_virtualspace.(Ref(ψ1), 1:L) == Vl₀
    end

    @testset "SU(2) Heisenberg" begin
        Random.seed!(577215)
        H = heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1 // 2, L)
        # SU(2) spin-1/2 bonds alternate between integer / half-integer spins, so use the
        # model's own full-rank virtual spaces rather than a hand-picked (integer-only) space.
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψ = FiniteMPS(physicalspace(H), maxV[2:(end - 1)])
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)
        Vl₀ = left_virtualspace.(Ref(ψ₀), 1:L)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, BUG())
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test left_virtualspace.(Ref(ψ1), 1:L) == Vl₀
    end

    # 3. Imaginary-time symmetric evolution lowers the energy while preserving the sector + norm.
    @testset "imaginary-time (U(1))" begin
        Random.seed!(141421)
        H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψi = FiniteMPS(physicalspace(H), maxV[2:(end - 1)])
        Vl₀ = left_virtualspace.(Ref(ψi), 1:L)
        Vr₀ = right_virtualspace.(Ref(ψi), 1:L)
        E_start = real(expectation_value(ψi, H))
        E_prev = E_start
        for _ in 1:15
            ψi, = timestep(ψi, H, 0.0, 0.1, BUG(); imaginary_evolution = true)
            E_now = real(expectation_value(ψi, H))
            @test E_now ≤ E_prev + 1.0e-6            # monotone (non-increasing) energy
            E_prev = E_now
        end
        @test E_prev < E_start - 0.5                 # substantial lowering toward the ground state
        @test norm(ψi) ≈ 1 atol = 1.0e-6             # imaginary-time BUG renormalizes each step
        # the sector content of every bond is preserved throughout the imaginary-time sweep
        @test left_virtualspace.(Ref(ψi), 1:L) == Vl₀
        @test right_virtualspace.(Ref(ψi), 1:L) == Vr₀
    end
end

# Charge-sector RANK-ADAPTIVE coverage (Chunk 2.3): the "sector-adaptivity" path — genuine
# symmetric tensors under a *truncating* `trscheme`, so bonds grow and shrink per sector. This is
# the hardest part of the design doc; it stresses the H2–H5/H10 pitfalls of the risk register:
# augmentation as a per-sector direct sum (H2/H10), the "old basis first" invariant per sector
# (H3), a global-ϑ truncation dropping a sector to dimension 0 ⇒ dynamic bond grading (H4), and
# total-boundary-charge conservation through the zero-block embeddings (H5). All states start
# genuinely low-rank so the dynamics *must* grow the bonds; no `force_planar`.
@testset "BUG rank-adaptive symmetric" verbose = true begin
    # A genuinely low-rank U(1) start. For spin-1/2 the virtual bonds alternate integer /
    # half-integer parity, so a single-sector cap collapses the state (no fusion channels); use a
    # mixed-parity cap with multiplicity 1 per sector. The full-rank Sz=0 profile is [1,2,4,8,4,2];
    # this cap gives [1,2,3,2,3,2], leaving room (esp. on the middle bond) for per-sector growth.
    u1cap = U1Space(-1 // 2 => 1, 1 // 2 => 1, 0 => 1, 1 => 1, -1 => 1)
    function low_rank_u1(L; seed = 2718, right = U1Space(0 => 1))
        Random.seed!(seed)
        H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
        ψ = FiniteMPS(physicalspace(H), u1cap; right)
        normalize!(ψ)
        return H, ψ
    end
    maxbond(ψ) = maximum(dim(left_virtualspace(ψ, k)) for k in 1:length(ψ))
    secmults(V) = [c => dim(V, c) for c in sectors(V)]

    # 1. THE CORE GATE: rank-adaptivity under symmetry grows the bond per sector for a tight ϑ and
    #    keeps it small for a loose ϑ, without any `SpaceMismatch`, while the total boundary charge
    #    (the fixed `right` virtual space) is preserved. Checked in the natural total-Sz=0 sector and
    #    in a fixed nonzero-charge (Sz=1) sector.
    @testset "rank-adaptivity + total-charge preservation (Sz = $label)" for (label, right) in
        (("0", U1Space(0 => 1)), ("1", U1Space(1 => 1)))
        L = 6
        H, ψ₀ = low_rank_u1(L; right)
        Rtot = right_virtualspace(ψ₀, L)
        Dstart = maxbond(ψ₀)

        ψtight = ψ₀
        for k in 0:2
            ψtight, = timestep(ψtight, H, k * 0.05, 0.05, BUG(; trscheme = truncerror(; atol = 1.0e-10)))
        end
        Dtight = maxbond(ψtight)

        ψloose = ψ₀
        for k in 0:2
            ψloose, = timestep(ψloose, H, k * 0.05, 0.05, BUG(; trscheme = truncerror(; atol = 1.0e-2)))
        end
        Dloose = maxbond(ψloose)

        @info "BUG rank-adaptive symmetric bond growth (Sz=$label)" Dstart Dloose Dtight
        @test Dtight > Dstart                            # per-sector augmentation grows the bond
        @test Dloose < Dtight                            # a looser tolerance keeps a smaller bond
        @test right_virtualspace(ψtight, L) == Rtot      # H5: total boundary charge preserved
        @test right_virtualspace(ψloose, L) == Rtot
        @test norm(ψtight) ≈ 1 atol = 1.0e-6             # tiny ϑ ⇒ negligible truncation
    end

    # 2. THE HARD GATE: overlap-error vs a *dense* `exp(-iH·T)` reference decreases as ϑ shrinks
    #    (monotonically up to the fixed-dt plateau). This proves per-sector augmentation+truncation
    #    actually captures the true dynamics under symmetry, not just that some bond grows.
    @testset "accuracy improves as ϑ decreases (U(1))" begin
        L = 6
        H, ψ₀ = low_rank_u1(L; seed = 202)

        Hmat = convert(TensorMap, H)
        ψvec = convert(TensorMap, ψ₀)
        ψvec /= norm(ψvec)

        T = 0.2
        dt = 0.05
        n = round(Int, T / dt)
        ref = exp(-im * Hmat * (n * dt)) * ψvec

        ϑs = [1.0e-2, 1.0e-4, 1.0e-6]
        errs = map(ϑs) do ϑ
            ψ = copy(ψ₀)
            for k in 0:(n - 1)
                ψ, = timestep(ψ, H, k * dt, dt, BUG(; trscheme = truncerror(; atol = ϑ)))
            end
            ψout = convert(TensorMap, ψ)
            ψout /= norm(ψout)
            return 1 - abs(dot(ψout, ref))
        end

        @info "BUG rank-adaptive symmetric accuracy vs ϑ" ϑs errs
        for i in 1:(length(ϑs) - 1)
            @test errs[i + 1] ≤ 1.5 * errs[i]   # monotone within plateau noise near the dt-floor
        end
        # Clear net improvement toward the dt-floor. This charge-capped low-rank system saturates
        # its (small) per-sector bonds quickly: at ϑ ≲ 1e-4 no directions are truncated, so the two
        # tightest tolerances land on the *identical* pure-dt-discretization floor (the ϑ=1e-4 and
        # 1e-6 errors coincide to the digit). The net truncation-limited improvement is therefore
        # modest (≈2.5×) rather than the orders of magnitude a full-rank trivial system shows.
        @test errs[end] < errs[1] / 2
    end

    # 3. DYNAMIC BOND GRADING: an interior bond's per-sector multiplicities
    #    `[dim(V, c) for c in sectors(V)]` are time-dependent — they change across a single
    #    rank-adaptive step (the graded structure is not fixed), while the total charge is.
    @testset "dynamic bond grading (per-sector multiplicities are time-dependent)" begin
        L = 6
        H, ψ₀ = low_rank_u1(L)
        mid = 4
        before = secmults(left_virtualspace(ψ₀, mid))
        ψ, = timestep(ψ₀, H, 0.0, 0.05, BUG(; trscheme = truncerror(; atol = 1.0e-10)))
        after = secmults(left_virtualspace(ψ, mid))
        @info "BUG rank-adaptive symmetric dynamic grading" before after
        @test before != after                                            # graded structure evolved
        @test right_virtualspace(ψ, L) == right_virtualspace(ψ₀, L)      # ... at fixed total charge
    end

    # 4. H4 — a global-ϑ cut can truncate a whole sector to dimension 0 (removing it from the bond),
    #    and downstream rank-adaptive steps must tolerate the changed / asymmetric grading.
    #
    #    NOTE on the sharpest H4 test (deterministic *drop-and-re-add* in one run): this is NOT
    #    achievable with single-site BUG, and asserting it would be wrong. The augmentation candidate
    #    `ACᵢ` in each half-sweep always carries the (already-truncated) bond as its *domain*, so
    #    `_bug_augment_left`/`_bug_augment_right` can only append directions whose charge sectors are
    #    already present on that bond — they grow multiplicity within existing sectors but can never
    #    re-introduce a sector that was truncated away (the graded analog of single-site TDVP's
    #    inability to change bond quantum numbers; re-adding a sector needs a two-site / CBE-style
    #    candidate, cf. `OptimalExpand`). We verified empirically that a sector dropped from an
    #    interior bond does not reappear over many subsequent tight steps. We therefore assert the
    #    deterministic *drop* and the H4 tolerance requirement (subsequent steps run without
    #    `SpaceMismatch`, yield a valid normalizable state, and preserve the total charge).
    @testset "sector drop-to-zero + dynamic-grading tolerance (H4)" begin
        L = 6
        H, ψ₀ = low_rank_u1(L)
        # grow a rich interior bond that carries the subdominant ±1 sectors
        ψrich = ψ₀
        for k in 0:5
            ψrich, = timestep(ψrich, H, k * 0.05, 0.05, BUG(; trscheme = truncerror(; atol = 1.0e-10)))
        end
        mid = 3
        Rtot = right_virtualspace(ψrich, L)
        @test U1Irrep(-1) in sectors(left_virtualspace(ψrich, mid))       # subdominant sector present

        # a global rank cut pools singular values across all sectors under one threshold, so the
        # subdominant sector is truncated to dimension 0 and removed from the bond (H4).
        ψdrop, = timestep(ψrich, H, 0.0, 0.05, BUG(; trscheme = truncrank(2)))
        @test !(U1Irrep(-1) in sectors(left_virtualspace(ψdrop, mid)))    # dropped to dim 0
        @test right_virtualspace(ψdrop, L) == Rtot                        # ... charge still preserved

        # subsequent rank-adaptive steps must tolerate the reduced / asymmetric grading: no
        # SpaceMismatch, a valid normalizable state, and a conserved total charge.
        ψcont = ψdrop
        for k in 0:3
            ψcont, = timestep(ψcont, H, k * 0.05, 0.05, BUG(; trscheme = truncerror(; atol = 1.0e-10)))
        end
        @info "BUG rank-adaptive symmetric H4" dropped = sectors(left_virtualspace(ψdrop, mid)) continued = sectors(left_virtualspace(ψcont, mid))
        @test isfinite(real(expectation_value(ψcont, H)))
        @test norm(ψcont) > 0
        @test right_virtualspace(ψcont, L) == Rtot
    end

    # 5. IMAGINARY-TIME symmetric rank-adaptive ground-state search: from a low bond dim it grows the
    #    per-sector bonds, lowers the energy toward `find_groundstate`, and preserves both the total
    #    charge and the (renormalized) norm.
    @testset "imaginary-time symmetric grows bond and lowers energy" begin
        L = 6
        H, ψ₀ = low_rank_u1(L)
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψgs, = find_groundstate(FiniteMPS(physicalspace(H), maxV[2:(end - 1)]), H; verbosity = 0)
        Egs = real(expectation_value(ψgs, H))

        Rtot = right_virtualspace(ψ₀, L)
        Dstart = maxbond(ψ₀)
        E_start = real(expectation_value(ψ₀, H))

        ψ = ψ₀
        for _ in 1:30
            ψ, = timestep(ψ, H, 0.0, 0.1, BUG(; trscheme = truncerror(; atol = 1.0e-8)); imaginary_evolution = true)
        end
        Dend = maxbond(ψ)
        E_end = real(expectation_value(ψ, H))

        @info "BUG rank-adaptive symmetric imaginary-time" Dstart Dend E_start E_end Egs
        @test Dend > Dstart                          # per-sector bonds grew as entanglement built up
        @test E_end < E_start - 1.0                  # substantial lowering
        @test E_end ≈ Egs atol = 0.1                 # toward the true ground state
        @test norm(ψ) ≈ 1 atol = 1.0e-6             # imaginary-time renormalizes each step
        @test right_virtualspace(ψ, L) == Rtot       # total charge preserved throughout
    end
end
