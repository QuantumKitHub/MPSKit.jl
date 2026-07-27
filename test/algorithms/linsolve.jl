println("
-----------------------------
|   Linear solver tests      |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
# KrylovKit is not a direct test dependency; reach the solver types through MPSKit's re-export
const GMRES = MPSKit.KrylovKit.GMRES
const CG = MPSKit.KrylovKit.CG
const BiCGStab = MPSKit.KrylovKit.BiCGStab
const DynamicTol = MPSKit.DynamicTol

verbosity_conv = 1

# helper: relative residual of (a₀ + a₁·A)·x = b, computed densely (robust for small systems;
# MPS-level `+`/`-` does not reliably cancel across states of differing bond structure)
function rel_residual(A, x, b, a₀, a₁)
    Am = convert(TensorMap, A)
    xv = convert(TensorMap, x)
    bv = convert(TensorMap, b)
    return norm(a₀ * xv + a₁ * (Am * xv) - bv) / norm(bv)
end

@testset "linsolve FiniteMPS" verbose = true begin
    L = 8
    H = force_planar(-transverse_field_ising(; L, g = -4))
    gs, = find_groundstate(FiniteMPS(L, ℙ^2, ℙ^16), H; verbosity = verbosity_conv)
    E₀ = real(expectation_value(gs, H))

    # resolvent of an eigenstate: (z − H)⁻¹|gs⟩ = 1/(z − E₀)|gs⟩, so ⟨gs|x⟩ = 1/(z − E₀).
    z = E₀ + 0.5 + 0.3im
    predicted = 1 / (z - E₀)

    @testset "resolvent, single-site, formulation $flav" for flav in (Galerkin(), LeastSquares())
        solver = flav isa LeastSquares ? CG(; tol = 1.0e-12) : GMRES(; tol = 1.0e-12)
        alg = DMRGSolve(; formulation = flav, solver, tol = 1.0e-10, verbosity = 0)
        # solve (z − H) x = gs  ⟺  a₀ = z, a₁ = −1
        x, envs, ϵ = linsolve(complex(copy(gs)), H, gs, alg; a₀ = z, a₁ = -1)
        @test dot(gs, x) ≈ predicted atol = 1.0e-6
        @test rel_residual(H, x, gs, z, -1) < 1.0e-6
    end

    @testset "resolvent, two-site (adaptive χ), formulation $flav" for flav in
        (Galerkin(), LeastSquares())
        solver = flav isa LeastSquares ? CG(; tol = 1.0e-12) : GMRES(; tol = 1.0e-12)
        alg = DMRGSolve2(;
            formulation = flav, solver, trunc = truncrank(16), tol = 1.0e-10, verbosity = 0
        )
        x, = linsolve(complex(copy(gs)), H, gs, alg; a₀ = z, a₁ = -1)
        @test dot(gs, x) ≈ predicted atol = 1.0e-5
        @test rel_residual(H, x, gs, z, -1) < 1.0e-5
    end

    @testset "matches propagator (correction vector)" begin
        alg_ls = DMRGSolve(; solver = GMRES(; tol = 1.0e-12), tol = 1.0e-10, verbosity = 0)
        x, = linsolve(complex(copy(gs)), H, gs, alg_ls; a₀ = z, a₁ = -1)
        for f in (NaiveInvert(), Jeckelmann())
            g_prop, = propagator(gs, z, H, DynamicalDMRG(; flavour = f, tol = 1.0e-10, verbosity = 0))
            @test dot(gs, x) ≈ g_prop atol = 1.0e-5
        end
    end

    @testset "adaptive tolerances (incl. BiCGStab)" begin
        # DynamicTol works for any solver, including short-recurrence BiCGStab (retunes only `tol`)
        alg_bicg = DMRGSolve(;
            solver = DynamicTol(BiCGStab(; tol = 1.0e-12); tol_min = 1.0e-14, tol_max = 1.0e-4),
            tol = 1.0e-9, verbosity = 0
        )
        x, = linsolve(complex(copy(gs)), H, gs, alg_bicg; a₀ = z, a₁ = -1)
        @test dot(gs, x) ≈ predicted atol = 1.0e-5
        @test rel_residual(H, x, gs, z, -1) < 1.0e-5

        # adaptive-by-default keyword path (DynamicTol-wrapped GMRES)
        x2, envs, ϵ = linsolve(complex(copy(gs)), H, gs; a₀ = z, a₁ = -1, tol = 1.0e-9, verbosity = 0)
        @test dot(gs, x2) ≈ predicted atol = 1.0e-5
        @test rel_residual(H, x2, gs, z, -1) < 1.0e-5
    end

    @testset "positive-definite solve via keyword interface" begin
        # s ≫ spectrum ⇒ (s − H) is positive-definite; the CG path is selected by `isposdef`
        s = E₀ + 50.0
        # eigenstate RHS: exact solution 1/(s − E₀) · gs, real and positive
        x, envs, ϵ = linsolve(copy(gs), H, gs; a₀ = s, a₁ = -1, isposdef = true, tol = 1.0e-10, verbosity = 0)
        @test dot(gs, x) ≈ 1 / (s - E₀) atol = 1.0e-8
        @test rel_residual(H, x, gs, s, -1) < 1.0e-8
    end

    @testset "general RHS, plain A·x = b against dense" begin
        # shift H to a well-conditioned, invertible operator A = s·I − H (s ≫ spectrum)
        s = E₀ + 50.0
        b = normalize!(FiniteMPS(randn, ComplexF64, L, ℙ^2, ℙ^8))
        alg = DMRGSolve2(;
            solver = CG(; tol = 1.0e-12), trunc = truncrank(64), tol = 1.0e-9, verbosity = 0
        ) & DMRGSolve(; solver = CG(; tol = 1.0e-12), tol = 1.0e-9, verbosity = 0)
        x, = linsolve(complex(copy(b)), H, b, alg; a₀ = s, a₁ = -1)
        @test rel_residual(H, x, b, s, -1) < 1.0e-4
    end
end

@testset "linsolve WindowMPS" verbose = true begin
    # `LeastSquares` builds the `A†b` term from the mixed sandwich ⟨x|A|b⟩, which is the only way to
    # reach it for a window operator (there is no `*(::WindowMPOHamiltonian, ::WindowMPS)`)
    N = 10
    H = transverse_field_ising(; g = -4)
    Ω, = find_groundstate(InfiniteMPS(ℂ^2, ℂ^10), H, VUMPS(; verbosity = 0))
    XΩ = WindowMPS(Ω, N)
    H_w = WindowMPOHamiltonian(H, 1:N)

    E₀ = expectation_value(XΩ, H_w)
    z = E₀ + 0.5 + 0.3im
    predicted = 1 / (z - E₀)

    @testset "resolvent, formulation $flav" for flav in (Galerkin(), LeastSquares())
        solver = flav isa LeastSquares ? CG(; tol = 1.0e-12) : GMRES(; tol = 1.0e-12)
        alg = DMRGSolve(; formulation = flav, solver, tol = 1.0e-10, verbosity = 0)
        x, = linsolve(XΩ, H_w, XΩ, alg; a₀ = z, a₁ = -1)
        @test dot(XΩ, x) ≈ predicted atol = 1.0e-6
    end
end
