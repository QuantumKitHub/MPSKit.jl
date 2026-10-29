using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Adapt
export algorithms_propagator_linear_solve_finitemps, algorithms_propagator_linear_solve_windowmps,
    algorithms_propagator_dynamical_dmrg_finitemps, algorithms_propagator_dynamical_dmrg_windowmps

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

function algorithms_propagator_linear_solve_finitemps(AT; fast = false)
    return @testset "linear_solve FiniteMPS ($AT)" verbose = true begin
        L = 8
        H = adapt(AT, force_planar(-transverse_field_ising(; L, g = -4)))
        gs, = find_groundstate(adapt(AT, FiniteMPS(L, ℙ^2, ℙ^16)), H; verbosity = verbosity_conv)
        E₀ = real(expectation_value(gs, H))

        # resolvent of an eigenstate: (z − H)⁻¹|gs⟩ = 1/(z − E₀)|gs⟩, so ⟨gs|x⟩ = 1/(z − E₀).
        z = E₀ + 0.5 + 0.3im
        predicted = 1 / (z - E₀)

        @testset "resolvent, single-site, formulation $flav" for flav in (Galerkin(), LeastSquares())
            solver = flav isa LeastSquares ? CG(; tol = 1.0e-12) : GMRES(; tol = 1.0e-12)
            alg = DMRGSolve(; formulation = flav, solver, tol = 1.0e-10, verbosity = 0)
            # solve (z − H) x = gs  ⟺  a₀ = z, a₁ = −1
            x, envs, info = linear_solve(complex(copy(gs)), H, gs, alg; a₀ = z, a₁ = -1)
            @test info.converged && convergence_measure(info) == info.galerkin
            @test dot(gs, x) ≈ predicted atol = 1.0e-6
            @test rel_residual(H, x, gs, z, -1) < 1.0e-6
        end

        @testset "resolvent, two-site (adaptive χ), formulation $flav" for flav in
            (Galerkin(), LeastSquares())
            solver = flav isa LeastSquares ? CG(; tol = 1.0e-12) : GMRES(; tol = 1.0e-12)
            alg = DMRGSolve2(;
                formulation = flav, solver, trunc = truncrank(16), tol = 1.0e-10, verbosity = 0
            )
            x, = linear_solve(complex(copy(gs)), H, gs, alg; a₀ = z, a₁ = -1)
            @test dot(gs, x) ≈ predicted atol = 1.0e-5
            @test rel_residual(H, x, gs, z, -1) < 1.0e-5
        end

        @testset "matches propagator (correction vector)" begin
            alg_ls = DMRGSolve(; solver = GMRES(; tol = 1.0e-12), tol = 1.0e-10, verbosity = 0)
            x, = linear_solve(complex(copy(gs)), H, gs, alg_ls; a₀ = z, a₁ = -1)
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
            x, = linear_solve(complex(copy(gs)), H, gs, alg_bicg; a₀ = z, a₁ = -1)
            @test dot(gs, x) ≈ predicted atol = 1.0e-5
            @test rel_residual(H, x, gs, z, -1) < 1.0e-5

            # adaptive-by-default keyword path (DynamicTol-wrapped GMRES)
            x2, = linear_solve(complex(copy(gs)), H, gs; a₀ = z, a₁ = -1, tol = 1.0e-9, verbosity = 0)
            @test dot(gs, x2) ≈ predicted atol = 1.0e-5
            @test rel_residual(H, x2, gs, z, -1) < 1.0e-5
        end

        @testset "positive-definite solve via keyword interface" begin
            # s ≫ spectrum ⇒ (s − H) is positive-definite; the CG path is selected by `isposdef`
            s = E₀ + 50.0
            # eigenstate RHS: exact solution 1/(s − E₀) · gs, real and positive
            x, = linear_solve(copy(gs), H, gs; a₀ = s, a₁ = -1, isposdef = true, tol = 1.0e-10, verbosity = 0)
            @test dot(gs, x) ≈ 1 / (s - E₀) atol = 1.0e-8
            @test rel_residual(H, x, gs, s, -1) < 1.0e-8
        end

        @testset "general RHS, plain A·x = b against dense" begin
            # shift H to a well-conditioned, invertible operator A = s·I − H (s ≫ spectrum)
            s = E₀ + 50.0
            b = normalize!(adapt(AT, FiniteMPS(randn, ComplexF64, L, ℙ^2, ℙ^8)))
            alg = DMRGSolve2(;
                solver = CG(; tol = 1.0e-12), trunc = truncrank(64), tol = 1.0e-9, verbosity = 0
            ) & DMRGSolve(; solver = CG(; tol = 1.0e-12), tol = 1.0e-9, verbosity = 0)
            x, = linear_solve(complex(copy(b)), H, b, alg; a₀ = s, a₁ = -1)
            @test rel_residual(H, x, b, s, -1) < 1.0e-4
        end
    end
end

function algorithms_propagator_linear_solve_windowmps(AT; fast = false)
    return @testset "linear_solve WindowMPS ($AT)" verbose = true begin
        # `LeastSquares` builds the `A†b` term from the mixed sandwich ⟨x|A|b⟩, which is the only way to
        # reach it for a window operator (there is no `*(::WindowMPOHamiltonian, ::WindowMPS)`)
        N = 10
        H = adapt(AT, transverse_field_ising(; g = -4))
        Ω, = find_groundstate(adapt(AT, InfiniteMPS(ℂ^2, ℂ^10)), H, VUMPS(; verbosity = 0))
        XΩ = WindowMPS(Ω, N)
        H_w = WindowMPOHamiltonian(H, 1:N)

        E₀ = expectation_value(XΩ, H_w)
        z = E₀ + 0.5 + 0.3im
        predicted = 1 / (z - E₀)

        @testset "resolvent, formulation $flav" for flav in (Galerkin(), LeastSquares())
            solver = flav isa LeastSquares ? CG(; tol = 1.0e-12) : GMRES(; tol = 1.0e-12)
            alg = DMRGSolve(; formulation = flav, solver, tol = 1.0e-10, verbosity = 0)
            x, = linear_solve(XΩ, H_w, XΩ, alg; a₀ = z, a₁ = -1)
            @test dot(XΩ, x) ≈ predicted atol = 1.0e-6
        end
    end
end

function algorithms_propagator_dynamical_dmrg_finitemps(AT; fast = false)
    return @testset "Dynamical DMRG FiniteMPS ($AT)" verbose = true begin
        L = 10
        H = adapt(AT, force_planar(-transverse_field_ising(; L, g = -4)))
        gs, = find_groundstate(adapt(AT, FiniteMPS(L, ℙ^2, ℙ^10)), H; verbosity = verbosity_conv)
        E₀ = expectation_value(gs, H)

        vals = (-0.5:0.2:0.5) .+ E₀
        eta = 0.3im

        predicted = [1 / (v + eta - E₀) for v in vals]

        @testset "Flavour $f" for f in (Jeckelmann(), NaiveInvert())
            alg = DynamicalDMRG(; flavour = f, verbosity = 0, tol = 1.0e-8)
            data = map(vals) do v
                result, = propagator(gs, v + eta, H, alg)
                return result
            end
            @test data ≈ predicted atol = 1.0e-8
        end

        # `trunc` prepends a two-site sweep, so the bond dimension of the initial guess need not
        # already be large enough: start from χ = 2 and let it grow back to the χ = 10 of the solution
        @testset "trunc (adaptive χ), flavour $f" for f in (Jeckelmann(), NaiveInvert())
            alg = DynamicalDMRG(;
                flavour = f, verbosity = 0, tol = 1.0e-8, trunc = truncrank(16)
            )
            init = adapt(AT, FiniteMPS(L, ℙ^2, ℙ^2))
            data = map(vals) do v
                result, = propagator(gs, v + eta, H, alg; init)
                return result
            end
            @test data ≈ predicted atol = 1.0e-6
        end
    end
end

function algorithms_propagator_dynamical_dmrg_windowmps(AT; fast = false)
    return @testset "Dynamical DMRG WindowMPS ($AT)" verbose = true begin
        N = 20

        H = adapt(AT, transverse_field_ising(g = -4))
        Ω = adapt(AT, InfiniteMPS(ComplexSpace(2), ComplexSpace(20)))

        (Ω, _) = find_groundstate(Ω, H, VUMPS(verbosity = verbosity_conv))
        XΩ = WindowMPS(Ω, N)
        H_w = WindowMPOHamiltonian(H, 1:N)

        gs_en = expectation_value(XΩ, H_w)
        vals = range(gs_en - 1.0, gs_en + 1.0, length = 5)
        eta = 0.3im
        predicted = [1 / (v + eta - gs_en) for v in vals]

        @testset "Flavour $f" for f in (Jeckelmann(), NaiveInvert())
            alg = DynamicalDMRG(; flavour = f, verbosity = 0, tol = 1.0e-8)
            data = map(vals) do v
                result, = propagator(XΩ, v + eta, H_w, alg)
                return result
            end
            @test data ≈ predicted atol = 1.0e-8
        end
    end
end
