
abstract type DDMRG_Flavour end

"
The 'original' flavor of dynamical DMRG, containing quadratic terms in (H-E)
https://arxiv.org/pdf/cond-mat/0203500.pdf

The Algorithm essentially minimizes
|| (H-E) |psi> - |target>||
"
struct Jeckelmann <: DDMRG_Flavour
end;

"
A simpler approach, with a less well defined approximation. We minmize : 

W(psi) = < psi | (H-E) | psi > - <psi | target> - <target | psi>

If we truly optimized the entire wavefunction psi, then we would obtain

|target> = (H-E) | psi >
"
struct NaiveInvert <: DDMRG_Flavour
end;

@kwdef struct DynamicalDMRG{F<:DDMRG_Flavour,S} <: Algorithm
    flavour::F = NaiveInvert

    solver::S = Defaults.linearsolver

    tol::Float64 = Defaults.tol * 10
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
end

function propagator(A::AbstractFiniteMPS, z, ham::MPOHamiltonian,
                    alg::DynamicalDMRG{NaiveInvert}; init=copy(A))
    h_envs = environments(init, ham) #environments for h
    mixedenvs = environments(init, A) #environments for <init | A>

    delta = 2 * alg.tol
    numit = 0

    while delta > alg.tol && numit < alg.maxiter
        numit += 1
        delta = 0.0

        for i in [1:(length(A) - 1); length(A):-1:2]
            tos = ac_proj(i, init, mixedenvs)

            H_AC = ∂∂AC(i, init, ham, h_envs)
            (res, convhist) = linsolve(H_AC, -tos, init.AC[i], alg.solver, -z, one(z))

            delta = max(delta, norm(res - init.AC[i]))
            init.AC[i] = res

            convhist.converged == 0 && @warn "($(i)) failed to converge $(convhist.normres)"
        end

        alg.verbose && @info "ddmrg sweep delta : $(delta)"
    end

    return dot(A, init), init
end

function propagator(A::AbstractFiniteMPS, z, ham::MPOHamiltonian,
                    alg::DynamicalDMRG{Jeckelmann}; init=copy(A))
    w = real(z)
    eta = imag(z)

    envs1 = environments(init, ham) #environments for h
    (ham2, envs2) = squaredenvs(init, ham, envs1) #environments for h^2
    mixedenvs = environments(init, A) #environments for <init | A>

    delta = 2 * alg.tol

    numit = 0
    while delta > alg.tol && numit < alg.maxiter
        numit += 1
        delta = 0.0

        for i in [1:(length(A) - 1); length(A):-1:2]
            tos = ac_proj(i, init, mixedenvs)
            #@plansor tos[-1 -2;-3] := leftenv(mixedenvs,i,init)[-1;1]*A.AC[i][1 -2;2]*rightenv(mixedenvs,i,init)[2;-3]

            H1_AC = ∂∂AC(i, init, ham, envs1)
            H2_AC = ∂∂AC(i, init, ham2, envs2)
            H_AC = LinearCombination((H1_AC, H2_AC), (-2 * w, 1))
            (res, convhist) = linsolve(H_AC, -eta * tos, init.AC[i], alg.solver,
                                       (eta * eta + w * w), 1)

            delta = max(delta, norm(res - init.AC[i]))
            init.AC[i] = res

            convhist.converged == 0 && @warn "($(i)) failed to converge $(convhist.normres)"
        end

        alg.verbose && @info "ddmrg sweep delta : $(delta)"
    end

    a = dot(ac_proj(1, init, mixedenvs), init.AC[1])
    #a = @plansor leftenv(mixedenvs,1,init)[-1;1]*A.AC[1][1 -2;2]*rightenv(mixedenvs,1,init)[2;-3]*conj(init.AC[1][-1 -2;-3])
    #a = a';

    cb = leftenv(envs1, 1, A) * TransferMatrix(init.AL, ham[1:length(A.AL)], A.AL)

    b = zero(a)
    for i in 1:length(cb)
        b += @plansor cb[i][1 2; 3] * init.CR[end][3; 4] *
                      rightenv(envs1, length(A), A)[i][4 2; 5] * conj(A.CR[end][1; 5])
    end

    v = b / eta - w / eta * a + 1im * a
    return v, init
end

function squaredenvs(state::AbstractFiniteMPS, ham::MPOHamiltonian,
                     envs=environments(state, ham))
    nham = conj(ham) * ham

    # to construct the squared caches we will first initialize environments
    # then make all data invalid so it will be recalculated
    # then initialize the right caches at the edge
    ncocache = environments(state, nham)

    # make sure the dependencies are incorrect, so data will be recalculated
    for i in 1:length(state)
        poison!(ncocache, i)
    end

    # impose the correct boundary conditions
    # (important for comoving mps, should do nothing for finite mps)
    indmap = LinearIndices((ham.odim, ham.odim))

    nleft = leftenv(ncocache, 1, state)
    nright = rightenv(ncocache, length(state), state)

    stor = storagetype(eltype(state.AL))
    for i in 1:(ham.odim), j in 1:(ham.odim)
        f1 = isomorphism(stor, space(nleft[indmap[i, j]], 2),
                         space(leftenv(envs, 1, state)[i], 2)' *
                         space(leftenv(envs, 1, state)[j], 2))
        @plansor begin
            nleft[indmap[i, j]][-1 -2; -3] := leftenv(envs, 1, state)[j][1 3; -3] *
                                              conj(leftenv(envs, 1, state)[i][1 2; -1]) *
                                              f1[-2; 2 3]
        end
        f2 = isomorphism(stor, space(nright[indmap[i, j]], 2),
                         space(rightenv(envs, length(state), state)[j], 2) *
                         space(rightenv(envs, length(state), state)[i], 2)')
        @plansor begin
            nright[indmap[i, j]][-1 -2; -3] := rightenv(envs, length(state), state)[j][-1 2;
                                                                                       1] *
                                               conj(rightenv(envs, length(state),
                                                             state)[i][-3 3; 1]) *
                                               f2[-2; 2 3]
        end
    end

    return nham, ncocache
end
