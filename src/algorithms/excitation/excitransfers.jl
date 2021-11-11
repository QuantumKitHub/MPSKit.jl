#excitation transfers - we default to regular transfers when no better candidate is found
exci_transfer_left(v,A,B=A) = transfer_left(v,A,B)
exci_transfer_right(v,A,B=A) = transfer_right(v,A,B)
exci_transfer_left(v,A,B,C) = transfer_left(v,A,B,C)
exci_transfer_right(v,A,B,C) = transfer_right(v,A,B,C)
exci_transfer_left(v,A,B,C,D) = transfer_left(v,A,B,C,D)
exci_transfer_right(v,A,B,C,D) = transfer_right(v,A,B,C,D)

#transfer, but the upper A is an excited tensor
exci_transfer_left(v::MPSBondTensor, A::MPOTensor, Ab::MPSTensor) =
    @plansor t[-1;-2 -3] := v[1;2]*A[2 3;-2 -3]*conj(Ab[1 3;-1])
exci_transfer_right(v::MPSBondTensor, A::MPOTensor, Ab::MPSTensor) =
    @plansor t[-1;-2 -3] := A[-1 3;-2 1]*v[1;2]*conj(Ab[-3 3;2])

#transfer, but the upper A is an excited tensor and there is an mpo leg being passed through
exci_transfer_left(v::MPSTensor, A::MPOTensor, Ab::MPSTensor) =
    @plansor t[-1 -2;-3 -4] := v[1 3;4]*A[4 5;-3 -4]*τ[3 2;5 -2]*conj(Ab[1 2;-1])

exci_transfer_right(v::MPSTensor, A::MPOTensor, Ab::MPSTensor) =
    @plansor t[-1 -2;-3 -4] := A[-1 4;-3 5]*τ[-2 3;4 2]*conj(Ab[-4 3;1])*v[5 2;1]


#mpo transfer, but with A an excitation-tensor
exci_transfer_left(v::MPSTensor,O::MPOTensor,A::MPOTensor,Ab::MPSTensor) =
    @plansor t[-1 -2;-3 -4] := v[4 2;1]*A[1 3;-3 -4]*O[2 5;3 -2]*conj(Ab[4 5;-1])
exci_transfer_right(v::MPSTensor,O::MPOTensor,A::MPOTensor,Ab::MPSTensor) =
    @plansor t[-1 -2;-3 -4] := A[-1 4;-3 5]*O[-2 2;4 3]*conj(Ab[-4 2;1])*v[5 3;1]

#mpo transfer, with an excitation leg
exci_transfer_left(v::MPOTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) =
    @plansor v[-1 -2;-3 -4] := v[4 2;-3 1]*A[1 3;-4]*O[2 5;3 -2]*conj(Ab[4 5;-1])
exci_transfer_right(v::MPOTensor,O::MPOTensor,A::MPSTensor,Ab::MPSTensor) =
    @plansor v[-1 -2;-3 -4] := A[-1 4;5]*O[-2 2;4 3]*conj(Ab[-4 2;1])*v[5 3;-3 1]

function exci_transfer_left(v,O::Vector{<:MPOTensor},A::Vector,Ab::Vector=A)
    for (o,a,ab) in zip(O,A,Ab)
        v = exci_transfer_left(v,o,a,ab)
    end
    v
end

function exci_transfer_right(v,O::Vector{<:MPOTensor},A::Vector,Ab::Vector=A)
    for (o,a,ab) in zip(reverse(O),reverse(A),reverse(Ab))
        v = exci_transfer_right(v,o,a,ab)
    end
    v
end

#A is an excitation tensor; with an excitation leg
exci_transfer_left(vec::Array{V,1},ham::SparseMPOSlice,A::M,Ab::V=A) where V<:MPSTensor where M <:MPOTensor =
    exci_transfer_left(M,vec,ham,A,Ab)
exci_transfer_right(vec::Array{V,1},ham::SparseMPOSlice,A::M,Ab::V=A) where V<:MPSTensor where M <:MPOTensor =
    exci_transfer_right(M,vec,ham,A,Ab)

#v has an extra excitation leg
exci_transfer_left(vec::Array{V,1},ham::SparseMPOSlice,A::M,Ab::M=A) where V<:MPOTensor where M <:MPSTensor =
    exci_transfer_left(V,vec,ham,A,Ab)
exci_transfer_right(vec::Array{V,1},ham::SparseMPOSlice,A::M,Ab::M=A) where V<:MPOTensor where M <:MPSTensor =
    exci_transfer_right(V,vec,ham,A,Ab)

function exci_transfer_left(RetType,vec,ham::SparseMPOSlice,A,Ab=A)
    toret = Array{RetType,1}(undef,length(vec));
    @sync for k in 1:ham.odim
        @Threads.spawn begin
            res = foldl(+, 1:ham.odim |>
                Filter(j->contains(ham,j,k)) |>
                Map() do j
                    if isscal(ham,j,k)
                        ham.Os[j,k]*exci_transfer_left(vec[j],A,Ab)
                    else
                        exci_transfer_left(vec[j],ham[j,k],A,Ab)
                    end
                end,init=Init(+));

            if res == Init(+)
                toret[k] = exci_transfer_left(vec[1],ham[1,k],A,Ab)
            else
                toret[k] = res;
            end
        end
    end
    toret
end
function exci_transfer_right(RetType,vec,ham::SparseMPOSlice,A,Ab=A)
    toret = Array{RetType,1}(undef,length(vec));

    @sync for j in 1:ham.odim
        @Threads.spawn begin
            res = foldl(+, 1:ham.odim |>
                Filter(k->contains(ham,j,k)) |>
                Map() do k
                    if isscal(ham,j,k)
                        ham.Os[j,k]*exci_transfer_right(vec[k],A,Ab)
                    else
                        exci_transfer_right(vec[k],ham[j,k],A,Ab)
                    end
                end,init=Init(+));
            if res == Init(+)
                toret[j] = exci_transfer_right(vec[1],ham[j,1],A,Ab)
            else
                toret[j] = res
            end
        end
    end
    toret
end

function left_excitation_transfer_system(lBs, ham, exci; mom=exci.momentum, solver=Defaults.linearsolver)
    len = ham.period
    found = zero.(lBs)
    ids = collect(Iterators.filter(x->isid(ham,x),1:ham.odim));

    for i in 1:ham.odim


        #this operation can be sped up by at least a factor 2;  found mostly consists of zeros
        start = found
        for k in 1:len
            start = exci_transfer_left(start,ham[k],exci.right_gs.AR[k],exci.left_gs.AL[k])*exp(conj(1im*mom))

            exci.trivial && for l in ids[2:end-1]
                @plansor start[l][-1 -2;-3 -4]-=start[l][1 4;-3 2]*r_RL(exci.right_gs,k)[2;3]*τ[3 4;5 1]*l_RL(exci.right_gs,k+1)[-1;6]*τ[5 6;-4 -2]
            end
        end

        #either the element i,i exists; in which case we have to solve a linear system
        #otherwise it's easy and we already know found[i]
        if reduce((a,b)->a&&contains(ham[b],i,i),1:len,init=true)
            (found[i],convhist) = linsolve(lBs[i]+start[i],lBs[i]+start[i],solver) do y
                x = @closure reduce(1:len,init=y) do a,b
                    exci_transfer_left(a,ham[b][i,i],exci.right_gs.AR[b],exci.left_gs.AL[b])*exp(conj(1im*mom))
                end

                if exci.trivial && i in ids
                    @plansor x[-1 -2;-3 -4] -= x[3 4;-3 5]*r_RL(exci.left_gs)[5;2]*τ[2 4;6 3]*l_RL(exci.left_gs)[-1;1]*τ[6 1;-4 -2]
                end

                return y-x
            end
            convhist.converged<1 && @info "left $(i) excitation inversion failed normres $(convhist.normres)"

        else
            found[i]=lBs[i]+start[i]
        end
    end
    return found
end

function right_excitation_transfer_system(rBs, ham, exci; mom=exci.momentum, solver=Defaults.linearsolver)
    len = ham.period
    found = zero.(rBs)
    ids = collect(Iterators.filter(x->isid(ham,x),1:ham.odim));
    for i in ham.odim:-1:1

        start = found
        for k in len:-1:1
            start = exci_transfer_right(start,ham[k],exci.left_gs.AL[k],exci.right_gs.AR[k])*exp(1im*mom)

            exci.trivial && for l in ids[2:end-1]
                @plansor start[l][-1 -2;-3 -4] -= τ[6 2;3 4]*start[l][3 4;-3 5]*l_LR(exci.right_gs,k)[5;2]*r_LR(exci.right_gs,k-1)[-1;1]*τ[-2 -4;1 6]
            end

        end

        if reduce((a,b)->a&&contains(ham[b],i,i),1:len,init=true)
            (found[i],convhist) = linsolve(rBs[i]+start[i],rBs[i]+start[i],solver) do y
                x = @closure reduce(len:-1:1,init=y) do a,b
                    exci_transfer_right(a,ham[b][i,i],exci.left_gs.AL[b],exci.right_gs.AR[b])*exp(1im*mom)
                end

                if exci.trivial && i in ids
                    @plansor x[-1 -2;-3 -4] -= τ[6 2;3 4]*x[3 4;-3 5]*l_LR(exci.right_gs)[5;2]*r_LR(exci.right_gs)[-1;1]*τ[-2 -4;1 6]
                end

                return y-x
            end
            convhist.converged<1 && @info "right $(i) excitation inversion failed normres $(convhist.normres)"

        else
            found[i]=rBs[i]+start[i]
        end
    end
    return found
end
