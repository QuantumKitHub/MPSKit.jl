#https://arxiv.org/pdf/1901.05824.pdf
#still need to add some tolerance - maxiter fields

struct WI <: Algorithm
end

struct WII <: Algorithm
end

function make_time_mpo(ham::MPOHamiltonian{S,T},dt,alg::WI) where {S,T}
    δ = dt*(-1im);
    W1 = PeriodicArray{T,3}(undef,ham.period,ham.odim-1,ham.odim-1);
    for loc in 1:size(ham,1)
        W1[loc,1,1] = ham[loc,1,1] + δ * ham[loc,1,ham.odim];

        for j in 2:size(ham,2)-1,
            k in 2:size(ham,3)-1
            W1[loc,j,k] = ham[loc,j,k]

            W1[loc,1,k] = ham[loc,1,k]*sqrt(δ)
            W1[loc,j,1] = ham[loc,j,ham.odim]*sqrt(δ)
        end
    end

    _makedense(W1)
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

        init = [one(ham[i,1,ham.odim]),zero(ham[i,1,k]),zero(ham[i,j,ham.odim]),zero(ham[i,j,k])]

        (y,convhist) = exponentiate(1.0,RecursiveVec(init),Lanczos()) do x
            out = similar(x.vecs);

            @tensor out[1][-1 -2;-3 -4] := δ*x[1][-1,1,-3,-4]*ham[i,1,ham.odim][2,-2,2,1]

            @tensor out[2][-1 -2;-3 -4] := δ*x[2][-1,1,-3,-4]*ham[i,1,ham.odim][2,-2,2,1]
            @tensor out[2][-1 -2;-3 -4] += sqrt(δ)*x[1][1,2,1,-4]*ham[i,1,k][-1,-2,-3,2]

            @tensor out[3][-1 -2;-3 -4] := δ*x[3][-1,1,-3,-4]*ham[i,1,ham.odim][2,-2,2,1]
            @tensor out[3][-1 -2;-3 -4] += sqrt(δ)*x[1][1,2,1,-4]*ham[i,j,ham.odim][-1,-2,-3,2]

            @tensor out[4][-1 -2;-3 -4] := δ*x[4][-1,1,-3,-4]*ham[i,1,ham.odim][2,-2,2,1]
            @tensor out[4][-1 -2;-3 -4] += x[1][1,2,1,-4]*ham[i,j,k][-1,-2,-3,2]
            @tensor out[4][-1 -2;-3 -4] += sqrt(δ)*x[2][1,2,-3,-4]*ham[i,j,ham.odim][-1,-2,1,2]
            @tensor out[4][-1 -2;-3 -4] += sqrt(δ)*x[3][-1,1,2,-4]*ham[i,1,k][2,-2,-3,1]

            RecursiveVec(out)
        end
        convhist.converged == 0 && @warn "failed to exponentiate $(convhist.normres)"
        
        WA[i,j-1,k-1] = y[4];
        WB[i,j-1] = y[3];
        WC[i,k-1] = y[2];
        WD[i] = y[1];
    end


    W2 = PeriodicArray{T,3}(undef,ham.period,ham.odim-1,ham.odim-1);
    W2[:,2:end,2:end] = WA
    W2[:,2:end,1] = WB
    W2[:,1,2:end] = WC
    W2[:,1,1] = WD

    for i in 1:ham.odim-1,
        j in 1:ham.odim-1,
        k in 1:ham.odim-1

        @assert space(W2[1,i,j],3) == space(W2[1,j,k],1)'
    end

    _makedense(W2)
end

function _makedense(ham)
    @assert length(size(ham)) == 3

    dense = PeriodicArray{eltype(ham),2}(undef,1,size(ham,1));

    domspaces = map(1:size(ham,1)) do loc
        [space(h,1) for h in ham[loc,:,1]]
    end

    embeds = PeriodicArray(_embedders.(domspaces));

    data = map(1:size(ham,1)) do loc
        reduce(+,map(Iterators.product(1:size(ham,2),1:size(ham,3))) do (i,j)
            @tensor temp[-1 -2;-3 -4]:=embeds[loc][i][-1,1]*ham[loc,i,j][1,-2,2,-4]*conj(embeds[loc+1][j][-3,2])
        end)
    end

    dense[1,:] = data
    PeriodicMPO(dense)
end
