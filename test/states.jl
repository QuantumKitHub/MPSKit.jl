println("------------------------------------")
println("|     States                       |")
println("------------------------------------")
@testset "FiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in [(𝔹^10, 𝔹^2, ComplexF64),
                                                             (Rep[SU₂](1 => 1, 0 => 3),
                                                              Rep[SU₂](0 => 1) * Rep[SU₂](0 => 1),
                                                              ComplexF32)]
    ts = FiniteMPS(rand, elt, rand(3:20), d, D)

    ovl = dot(ts, ts)

    @test ovl ≈ norm(ts.AC[1])^2

    for i in 1:length(ts)
        @test ts.AC[i] ≈ ts.AL[i] * ts.CR[i]
        @test ts.AC[i] ≈
              MPSKit._transpose_front(ts.CR[i - 1] * MPSKit._transpose_tail(ts.AR[i]))
    end

    @test elt == eltype(eltype(ts))

    ts = ts * 3
    @test ovl * 9 ≈ norm(ts)^2
    ts = 3 * ts
    @test ovl * 9 * 9 ≈ norm(ts)^2

    @test norm(2 * ts + ts - 3 * ts) ≈ 0.0 atol = sqrt(eps(real(elt)))
end

@testset "FiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in [(𝔹^10, 𝔹^2, ComplexF64),
                                                             (Rep[U₁](-1 => 3, 0 => 3, 1 => 3),
                                                              Rep[U₁](-1 => 1, 0 => 1, 1 => 1),
                                                              ComplexF64)]
    ts_small = FiniteMPS(rand, elt, 4, d, D)
    ts_small2 = FiniteMPS(MPSKit.decompose_localmps(convert(TensorMap, ts_small)))
    @test dot(ts_small, ts_small2) ≈ dot(ts_small, ts_small)
end

@testset "InfiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in [(𝔹^10, 𝔹^2, ComplexF64),
                                                               (Rep[U₁](1 => 3), Rep[U₁](0 => 1),
                                                                ComplexF64)]
    tol = Float64(eps(real(elt)) * 100)

    ts = InfiniteMPS([TensorMap(rand, elt, D * d, D), TensorMap(rand, elt, D * d, D)];
                     tol=tol)

    for i in 1:length(ts)
        @plansor difference[-1 -2; -3] := ts.AL[i][-1 -2; 1] * ts.CR[i][1; -3] -
                                          ts.CR[i - 1][-1; 1] * ts.AR[i][1 -2; -3]
        @test norm(difference, Inf) < tol * 10

        @test l_LL(ts, i) * TransferMatrix(ts.AL[i], ts.AL[i]) ≈ l_LL(ts, i + 1)
        @test l_LR(ts, i) * TransferMatrix(ts.AL[i], ts.AR[i]) ≈ l_LR(ts, i + 1)
        @test l_RL(ts, i) * TransferMatrix(ts.AR[i], ts.AL[i]) ≈ l_RL(ts, i + 1)
        @test l_RR(ts, i) * TransferMatrix(ts.AR[i], ts.AR[i]) ≈ l_RR(ts, i + 1)

        @test TransferMatrix(ts.AL[i], ts.AL[i]) * r_LL(ts, i) ≈ r_LL(ts, i + 1)
        @test TransferMatrix(ts.AL[i], ts.AR[i]) * r_LR(ts, i) ≈ r_LR(ts, i + 1)
        @test TransferMatrix(ts.AR[i], ts.AL[i]) * r_RL(ts, i) ≈ r_RL(ts, i + 1)
        @test TransferMatrix(ts.AR[i], ts.AR[i]) * r_RR(ts, i) ≈ r_RR(ts, i + 1)
    end
end

@testset "MPSMultiline ($(sectortype(D)), $elt)" for (D, d, elt) in
                                                     [(𝔹^10, 𝔹^2, ComplexF64),
                                                                (Rep[U₁](1 => 3), Rep[U₁](0 => 1),
                                                                 ComplexF32)]
    tol = Float64(eps(real(elt)) * 100)
    ts = MPSMultiline([TensorMap(rand, elt, D * d, D) TensorMap(rand, elt, D * d, D);
                       TensorMap(rand, elt, D * d, D) TensorMap(rand, elt, D * d, D)];
                      tol=tol)

    for i in 1:size(ts, 1), j in 1:size(ts, 2)
        @plansor difference[-1 -2; -3] := ts.AL[i, j][-1 -2; 1] * ts.CR[i, j][1; -3] -
                                          ts.CR[i, j - 1][-1; 1] * ts.AR[i, j][1 -2; -3]
        @test norm(difference, Inf) < tol * 10

        @test l_LL(ts, i, j) * TransferMatrix(ts.AL[i, j], ts.AL[i, j]) ≈ l_LL(ts, i, j + 1)
        @test l_LR(ts, i, j) * TransferMatrix(ts.AL[i, j], ts.AR[i, j]) ≈ l_LR(ts, i, j + 1)
        @test l_RL(ts, i, j) * TransferMatrix(ts.AR[i, j], ts.AL[i, j]) ≈ l_RL(ts, i, j + 1)
        @test l_RR(ts, i, j) * TransferMatrix(ts.AR[i, j], ts.AR[i, j]) ≈ l_RR(ts, i, j + 1)

        @test TransferMatrix(ts.AL[i, j], ts.AL[i, j]) * r_LL(ts, i, j) ≈ r_LL(ts, i, j + 1)
        @test TransferMatrix(ts.AL[i, j], ts.AR[i, j]) * r_LR(ts, i, j) ≈ r_LR(ts, i, j + 1)
        @test TransferMatrix(ts.AR[i, j], ts.AL[i, j]) * r_RL(ts, i, j) ≈ r_RL(ts, i, j + 1)
        @test TransferMatrix(ts.AR[i, j], ts.AR[i, j]) * r_RR(ts, i, j) ≈ r_RR(ts, i, j + 1)
    end
end

@testset "WindowMPS" begin
    ham = force_planar(transverse_field_ising(; g=8.0))
    (gs, _, _) = find_groundstate(InfiniteMPS([𝔹^2], [𝔹^10]), ham, VUMPS(; verbose=false))

    #constructor 1 - give it a plain array of tensors
    window_1 = WindowMPS(gs, copy.([gs.AC[1]; [gs.AR[i] for i in 2:10]]), gs)

    #constructor 2 - used to take a "slice" from an infinite mps
    window_2 = WindowMPS(gs, 10)

    # we should logically have that window_1 approximates window_2
    ovl = dot(window_1, window_2)
    @test ovl ≈ 1 atol = 1e-8

    #constructor 3 - random initial tensors
    window = WindowMPS(rand, ComplexF64, 10, 𝔹^2, 𝔹^10, gs, gs)
    normalize!(window)
    
    for i in 1:length(window)
        @test window.AC[i] ≈ window.AL[i] * window.CR[i]
        @test window.AC[i] ≈ MPSKit._transpose_front(window.CR[i - 1] *
                                                     MPSKit._transpose_tail(window.AR[i]))
    end

    @test norm(window) ≈ 1
    window = window * 3
    @test 9 ≈ norm(window)^2
    window = 3 * window
    @test 9 * 9 ≈ norm(window)^2
    normalize!(window)

    e1 = expectation_value(window, ham)

    v1 = variance(window, ham)
    (window, envs, _) = find_groundstate(window, ham, DMRG(; verbose=false))
    v2 = variance(window, ham)

    e2 = expectation_value(window, ham)

    @test v2 < v1
    @test real(e2[2]) ≤ real(e1[2])

    (window, envs) = timestep(window, ham, 0.1, TDVP2(), envs)
    (window, envs) = timestep(window, ham, 0.1, TDVP(), envs)

    e3 = expectation_value(window, ham)

    #why is this not exactly the same anymore? TDVP() is fine, TDVP2() make difference of the order 1e-06
    @test real.(e2[1]) ≈ real.(e3[1]) atol = 1e-04
    @test real(e2[2]) ≈ real(e3[2]) atol = 1e-04
end

@testset "Quasiparticle state" verbose=true begin
    @testset "Finite" verbose=true for (th, D, d) in
                               [(force_planar(transverse_field_ising()), 𝔹^10, 𝔹^2),
                                (heisenberg_XXX(SU2Irrep; spin=1), Rep[SU₂](1 => 1, 0 => 3),
                                 Rep[SU₂](1 => 1))]
        ts = FiniteMPS(rand, ComplexF64, rand(4:20), d, D)
        normalize!(ts)

        #rand_quasiparticle is a private non-exported function
        qst1 = MPSKit.LeftGaugedQP(rand, ts)
        qst2 = MPSKit.LeftGaugedQP(rand, ts)

        @test norm(axpy!(1, qst1, copy(qst2))) ≤ norm(qst1) + norm(qst2)
        @test norm(qst1) * 3 ≈ norm(qst1 * 3)

        normalize!(qst1)

        qst1_f = convert(FiniteMPS, qst1)
        qst2_f = convert(FiniteMPS, qst2)

        ovl_f = dot(qst1_f, qst2_f)
        ovl_q = dot(qst1, qst2)
        @test ovl_f ≈ ovl_q atol = 1e-5
        @test norm(qst1_f) ≈ norm(qst1) atol = 1e-5

        
        
        ev_f = sum(expectation_value(qst1_f, th) - expectation_value(ts, th))

       
        ev_q = dot(qst1, effective_excitation_hamiltonian(th, qst1))
        @test ev_f ≈ ev_q atol = 1e-5

    end

    @testset "Infinite" for (th, D, d) in
                                 [(force_planar(transverse_field_ising()), 𝔹^10, 𝔹^2),
                                  (heisenberg_XXX(SU2Irrep; spin=1), Rep[SU₂](1 => 1, 0 => 3),
                                   Rep[SU₂](1 => 1))]
        period = rand(1:4)
        ts = InfiniteMPS(fill(d, period), fill(D, period))

        #rand_quasiparticle is a private non-exported function
        qst1 = MPSKit.LeftGaugedQP(rand, ts)
        qst2 = MPSKit.LeftGaugedQP(rand, ts)

        @test norm(axpy!(1, qst1, copy(qst2))) ≤ norm(qst1) + norm(qst2)
        @test norm(qst1) * 3 ≈ norm(qst1 * 3)

        @test dot(qst1, convert(MPSKit.LeftGaugedQP, convert(MPSKit.RightGaugedQP, qst1))) ≈
              dot(qst1, qst1) atol = 1e-10
    end
end

@testset "Copy $(d)" for (D,d) in [(𝔹^10, 𝔹^2),
                                (Rep[SU₂](1 => 1, 0 => 3),Rep[SU₂](1 => 1)),
                                (Rep[U₁]((0 => 20)), Rep[U₁](0 => 2))]
    @testset "InfiniteMPS $(d)" begin

        period = rand(1:4)
        Ψ = InfiniteMPS(fill(d, period), fill(D, period))
        Ψ_copied = copy(Ψ);

        norm(Ψ)
        Ψ.AC[1] *= 2;
        norm(Ψ)
        @test abs(norm(Ψ_copied) - norm(Ψ)) > 0.5
    end

    @testset "WindowMPS $(d)" begin

        period = rand(1:4)
        Ψ = InfiniteMPS(fill(d, period), fill(D, period));

        Ψwindow = WindowMPS(rand, ComplexF64, rand(5:10), d, D, Ψ, Ψ);

        @test Ψwindow.left_gs !== Ψwindow.right_gs # not the same reference
        @test Ψwindow.left_gs ≈ Ψwindow.right_gs  # but the same state

        Ψwindow.left_gs.AC[1] *= 2;
        @test abs(norm(Ψwindow.left_gs) - norm(Ψwindow.right_gs)) > 0.5
        @test abs(norm(Ψwindow.left_gs) - norm(Ψ)) > 0.5

        Ψwindow = WindowMPS(rand, ComplexF64, rand(5:10), d, D, Ψ, Ψ);
        Ψwindow_copied = copy(Ψwindow);

        Ψwindow.left_gs.AC[1] *= 2;
        @test abs(norm(Ψwindow_copied.left_gs) - norm(Ψwindow.left_gs)) > 0.5

        Ψwindow.right_gs.AC[1] *= 2;
        @test abs(norm(Ψwindow_copied.right_gs) - norm(Ψwindow.right_gs)) > 0.5

        Ψwindow.window.AC[1] *= 2;
        @test abs(norm(Ψwindow_copied.window) - norm(Ψwindow.window)) > 0.5
    end
end