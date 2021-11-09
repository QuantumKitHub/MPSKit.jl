#https://arxiv.org/pdf/1901.05824.pdf

struct WI <: Algorithm
end

@with_kw struct WII <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

function make_time_mpo(ham::MPOHamiltonian{S,T},dt,alg::WI) where {S,T}
    δ = dt*(-1im);
    W1 = PeriodicArray{Union{T,Missing},3}(missing,ham.period,ham.odim-1,ham.odim-1);
    for loc in 1:size(ham,1)
        W1[loc,1,1] = ham[loc][1,1] + δ * ham[loc][1,ham.odim];

        for j in 2:size(ham,2)-1,
            k in 2:size(ham,3)-1
            W1[loc,j,k] = ham[loc][j,k]

            W1[loc,1,k] = ham[loc][1,k]*sqrt(δ)
            W1[loc,j,1] = ham[loc][j,ham.odim]*sqrt(δ)
        end
    end

    SparseMPO(W1)
end


function make_time_mpo(ham::MPOHamiltonian{S,T},dt,alg::WII) where {S,T}
    WA = PeriodicArray{T,3}(undef,ham.period,ham.odim-2,ham.odim-2);
    WB = PeriodicArray{T,2}(undef,ham.period,ham.odim-2);
    WC = PeriodicArray{T,2}(undef,ham.period,ham.odim-2);
    WD = PeriodicArray{T,1}(undef,ham.period);

    δ = dt*(-1im);

    for i in 1:ham.period,
        j in 2:ham.odim-1,
        k in 2:ham.odim-1

        init_1 = isometry(storagetype(ham[i][1,ham.odim]),codomain(ham[i][1,ham.odim]),domain(ham[i][1,ham.odim]))
        init = [init_1,zero(ham[i][1,k]),zero(ham[i][j,ham.odim]),zero(ham[i][j,k])]

        (y,convhist) = exponentiate(1.0,RecursiveVec(init),Arnoldi(tol = alg.tol,maxiter = alg.maxiter)) do x
            out = similar(x.vecs);

            @plansor out[1][-1 -2;-3 -4] := δ*x[1][-1 1;-3 -4]*ham[i][1,ham.odim][2 3;1 4]*τ[-2 4;2 3]

            @plansor out[2][-1 -2;-3 -4] := δ*x[2][-1 1;-3 -4]*ham[i][1,ham.odim][2 3;1 4]*τ[-2 4;2 3]
            @plansor out[2][-1 -2;-3 -4] += sqrt(δ)*x[1][1 2;-3 4]*ham[i][1,k][-1 -2;3 -4]*τ[3 4;1 2]

            @plansor out[3][-1 -2;-3 -4] := δ*x[3][-1 1;-3 -4]*ham[i][1,ham.odim][2 3;1 4]*τ[-2 4;2 3]
            @plansor out[3][-1 -2;-3 -4] += sqrt(δ)*x[1][1 2;-3 4]*ham[i][j,ham.odim][-1 -2;3 -4]*τ[3 4;1 2]

            @plansor out[4][-1 -2;-3 -4] := δ*x[4][-1 1;-3 -4]*ham[i][1,ham.odim][2 3;1 4]*τ[-2 4;2 3]
            @plansor out[4][-1 -2;-3 -4] += x[1][1 2;-3 4]*ham[i][j,k][-1 -2;3 -4]*τ[3 4;1 2]
            @plansor out[4][-1 -2;-3 -4] += sqrt(δ)*x[2][1 2;-3 -4]*ham[i][j,ham.odim][-1 -2;3 4]*τ[3 4;1 2]
            @plansor out[4][-1 -2;-3 -4] += sqrt(δ)*x[3][-1 4;-3 3]*ham[i][1,k][2 -2;1 -4]*τ[3 4;1 2]

            RecursiveVec(out)
        end
        convhist.converged == 0 && @warn "failed to exponentiate $(convhist.normres)"

        WA[i,j-1,k-1] = y[4];
        WB[i,j-1] = y[3];
        WC[i,k-1] = y[2];
        WD[i] = y[1];
    end


    W2 = PeriodicArray{Union{T,Missing},3}(missing,ham.period,ham.odim-1,ham.odim-1);
    W2[:,2:end,2:end] = WA
    W2[:,2:end,1] = WB
    W2[:,1,2:end] = WC
    W2[:,1,1] = WD

    SparseMPO(W2)
end
