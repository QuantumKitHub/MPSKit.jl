#=
nothing fancy - only used internally (and therefore cryptic) - stores some partially contracted things
seperates out this bit of logic from effective_excitation_hamiltonian (now more readable)
can also - potentially - partially reuse this in other algorithms
=#
struct QPEnv{A,B} <: Cache
    lBs::PeriodicArray{A,1}
    rBs::PeriodicArray{A,1}

    lenvs::B
    renvs::B
end

function environments(exci::Union{InfiniteQP,Multiline{<:InfiniteQP}}, H; solver=Defaults.linearsolver)
    # Explicitly define optional arguments as these depend on solver,
    # which needs to come after these arguments.
    lenvs = environments(exci.left_gs, H; solver=solver)

    return environments(exci, H, lenvs; solver=solver)
end

function environments(exci::Union{InfiniteQP,Multiline{<:InfiniteQP}}, H, lenvs; solver=Defaults.linearsolver)
    # Explicitly define optional arguments as these depend on solver,
    # which needs to come after these arguments.
    renvs = exci.trivial ? lenvs : environments(exci.right_gs, H; solver=solver)

    return environments(exci, H, lenvs, renvs; solver=solver)
end

function environments(exci::InfiniteQP, ham::MPOHamiltonian, lenvs, renvs;solver=Defaults.linearsolver)
    ids = collect(Iterators.filter(x->isid(ham,x),2:ham.odim-1));

    AL = exci.left_gs.AL;
    AR = exci.right_gs.AR;

    #build lBs(c)
    lB_cur = [ TensorMap(zeros,eltype(exci),
                    left_virtualspace(exci.left_gs,0)*ham.domspaces[1,k]',
                    space(exci[1],3)'*right_virtualspace(exci.right_gs,0)) for k in 1:ham.odim]
    lBs = typeof(lB_cur)[]

    for pos in  1:length(exci)
        lB_cur = lB_cur * TransferMatrix(AR[pos],ham[pos],AL[pos])*exp(-1im*exci.momentum)
        lB_cur += leftenv(lenvs,pos,exci.left_gs) * TransferMatrix(exci[pos],ham[pos],AL[pos])*exp(-1im*exci.momentum)

        exci.trivial && for i in ids
            @plansor lB_cur[i][-1 -2;-3 -4] -= lB_cur[i][1 4;-3 2]*r_RL(exci.left_gs,pos)[2;3]*τ[3 4;5 1]*l_RL(exci.left_gs,pos+1)[-1;6]*τ[5 6;-4 -2]
        end


        push!(lBs,lB_cur)
    end

    #build rBs(c)
    rB_cur = [ TensorMap(zeros,eltype(exci),
                    left_virtualspace(exci.left_gs,length(exci))*ham.imspaces[length(exci),k]',
                    space(exci[1],3)'*right_virtualspace(exci.right_gs,length(exci))) for k in 1:ham.odim]
    rBs = typeof(rB_cur)[]

    for pos in length(exci):-1:1
        rB_cur = TransferMatrix(AL[pos],ham[pos],AR[pos])*rB_cur*exp(1im*exci.momentum)
        rB_cur += TransferMatrix(exci[pos],ham[pos],AR[pos])*rightenv(renvs,pos,exci.right_gs)*exp(1im*exci.momentum)

        exci.trivial && for i in ids
            @plansor rB_cur[i][-1 -2;-3 -4] -= τ[6 4;1 3]*rB_cur[i][1 3;-3 2]*l_LR(exci.left_gs,pos)[2;4]*r_LR(exci.left_gs,pos-1)[-1;5]*τ[-2 -4;5 6]
        end

        push!(rBs,rB_cur)
    end
    rBs = reverse(rBs)

    local lBE::typeof(rB_cur);
    local rBE::typeof(rB_cur);

    @sync begin
        @Threads.spawn lBE = left_excitation_transfer_system($lB_cur,$ham,$exci,solver=$solver)
        @Threads.spawn rBE = right_excitation_transfer_system($rB_cur,$ham,$exci, solver=$solver)
    end

    lBs[end] = lBE;

    for i=1:length(exci)-1
        lBE = lBE*TransferMatrix(AR[i],ham[i],AL[i])*exp(-1im*exci.momentum)

        exci.trivial && for k in ids
            @plansor lBE[k][-1 -2;-3 -4] -= lBE[k][1 4;-3 2]*r_RL(exci.left_gs,i)[2;3]*τ[3 4;5 1]*l_RL(exci.left_gs,i+1)[-1;6]*τ[5 6;-4 -2]
        end

        lBs[i] += lBE;
    end

    rBs[1] = rBE;

    for i=length(exci):-1:2
        rBE = TransferMatrix(AL[i],ham[i],AR[i])*rBE*exp(1im*exci.momentum)

        exci.trivial && for k in ids
            @plansor rBE[k][-1 -2;-3 -4] -= τ[6 4;1 3]*rBE[k][1 3;-3 2]*l_LR(exci.left_gs,i)[2;4]*r_LR(exci.left_gs,i-1)[-1;5]*τ[-2 -4;5 6]
        end

        rBs[i] += rBE
    end

    return QPEnv(PeriodicArray(lBs),PeriodicArray(rBs),lenvs,renvs)
end

function environments(exci::FiniteQP,ham::MPOHamiltonian,lenvs=environments(exci.left_gs,ham),renvs=exci.trivial ? lenvs : environments(exci.right_gs,ham))

    AL = exci.left_gs.AL;
    AR = exci.right_gs.AR;

    #construct lBE
    lB_cur = [ TensorMap(zeros,eltype(exci),
                    left_virtualspace(exci.left_gs,0)*ham.domspaces[1,k]',
                    space(exci[1],3)'*right_virtualspace(exci.left_gs,0)) for k in 1:ham.odim]
    lBs = typeof(lB_cur)[]
    for pos = 1:length(exci)
        lB_cur = lB_cur*TransferMatrix(AR[pos],ham[pos],AL[pos])
        lB_cur += leftenv(lenvs,pos,exci.left_gs)*TransferMatrix(exci[pos],ham[pos],AL[pos])
        push!(lBs,lB_cur)
    end

    #build rBs(c)
    rB_cur = [ TensorMap(zeros,eltype(exci),
                    left_virtualspace(exci.right_gs,length(exci))*ham.imspaces[length(exci),k]',
                    space(exci[1],3)'*right_virtualspace(exci.right_gs,length(exci))) for k in 1:ham.odim]
    rBs = typeof(rB_cur)[]
    for pos=length(exci):-1:1
        rB_cur = TransferMatrix(AL[pos],ham[pos],AR[pos])*rB_cur
        rB_cur += TransferMatrix(exci[pos],ham[pos],AR[pos])*rightenv(renvs,pos,exci.right_gs)
        push!(rBs,rB_cur)
    end
    rBs=reverse(rBs)

    return QPEnv(PeriodicArray(lBs),PeriodicArray(rBs),lenvs,renvs)
end

function environments(exci::Multiline{<:InfiniteQP}, ham::MPOMultiline, lenvs, renvs;solver=Defaults.linearsolver)
    exci.trivial || @warn "there is a phase ambiguity in topologically nontrivial statmech excitations"

    left_gs = exci.left_gs;
    right_gs = exci.right_gs;

    AL = left_gs.AL;
    AR = right_gs.AR;

    exci_space = space(exci[1][1],3);

    lBs = map(1:size(left_gs,1)) do row
        renorms = (map(1:size(left_gs,2)) do col
            v = leftenv(lenvs,row,col,left_gs)*TransferMatrix(left_gs.AC[row,col],ham[row,col],left_gs.AC[row+1,col]);
            @plansor v[1 2;3]*rightenv(lenvs,row,col,left_gs)[3 2;1]
        end).^-1;

        lB_cur = TensorMap(zeros,eltype(AL[1,1]),
                                left_virtualspace(left_gs,row+1,0)*_firstspace(ham[row,1])',
                                exci_space'*right_virtualspace(right_gs,row,0));

        c_lBs = typeof(lB_cur)[];
        for col in 1:size(left_gs,2)
            lB_cur = renorms[col]*lB_cur*TransferMatrix(AR[row,col],ham[row,col],AL[row+1,col])*exp(-1im*exci.momentum)
            lB_cur += renorms[col]*leftenv(lenvs,row,col,left_gs)*TransferMatrix(exci[row][col],ham[row,col],AL[row+1,col])*exp(-1im*exci.momentum)
            push!(c_lBs,lB_cur);
        end

        tm = TransferMatrix(AR[row,:],ham[row,:],AL[row+1,:]);

        (c_lBs[end],convhist) = linsolve(lB_cur,lB_cur,solver,1,-exp(-1im*size(left_gs,2)*exci.momentum)*prod(renorms)) do v
            y = v*tm;
            if exci.trivial
                @plansor y[-1 -2;-3 -4] -= y[4 2;-3 3]*conj(left_gs.CR[row+1,0][4;1])*rightenv(lenvs,row,0,left_gs)[3 2;1]*leftenv(lenvs,row,1,left_gs)[-1 -2;5]*left_gs.CR[row,0][5;-4]
            end
            y
        end
        convhist.converged == 0 && @warn "lbe failed to converge $(convhist.normres)"

        cur = c_lBs[end];
        for col in 1:size(left_gs,2)-1
            cur = renorms[col]*cur*TransferMatrix(AR[row,col],ham[row,col],AL[row+1,col])*exp(conj(1im*exci.momentum))
            c_lBs[col] += cur
        end

        PeriodicArray(c_lBs)
    end

    rBs = map(1:size(left_gs,1)) do row
        renorms = (map(1:size(right_gs,2)) do col
            v = leftenv(renvs,row,col,right_gs)*TransferMatrix(right_gs.AC[row,col],ham[row,col],right_gs.AC[row+1,col]);
            @plansor v[1 2;3]*rightenv(renvs,row,col,right_gs)[3 2;1]
        end).^-1;

        rB_cur = TensorMap(zeros,eltype(AL[1,1]),
                                left_virtualspace(left_gs,row,0)*_firstspace(ham[row,1]),
                                exci_space'*right_virtualspace(right_gs,row+1,0));

        c_rBs = typeof(rB_cur)[];
        for col in size(left_gs,2):-1:1
            rB_cur = TransferMatrix(AL[row,col],ham[row,col],AR[row+1,col])*rB_cur*exp(1im*exci.momentum)*renorms[col]
            rB_cur += TransferMatrix(exci[row][col],ham[row,col],AR[row+1,col])*rightenv(renvs,row,col,right_gs)*exp(1im*exci.momentum)*renorms[col]
            push!(c_rBs,rB_cur);
        end
        c_rBs = reverse(c_rBs);

        tm = TransferMatrix(AL[row,:],ham[row,:],AR[row+1,:])
        (c_rBs[1],convhist) = linsolve(rB_cur,rB_cur,GMRES(),1,-exp(1im*size(left_gs,2)*exci.momentum)*prod(renorms)) do v
            y = tm*v;
            if exci.trivial
                @plansor y[-1 -2;-3 -4] -= y[1 2;-3 3]*conj(left_gs.CR[row+1,0][1;4])*leftenv(lenvs,row,1,left_gs)[3 2;4]*rightenv(lenvs,row,0,left_gs)[5 -2;-4]*left_gs.CR[row,0][-1;5]
            end
            y
        end
        convhist.converged == 0 && @warn "rbe failed to converge $(convhist.normres)"

        cur = c_rBs[1];
        for col in size(left_gs,2):-1:2
            cur = TransferMatrix(AL[row,col],ham[row,col],AR[row+1,col])*cur*exp(1im*exci.momentum)*renorms[col]
            c_rBs[col] += cur
        end

        PeriodicArray(c_rBs)
    end

    QPEnv(PeriodicArray(lBs),PeriodicArray(rBs),lenvs,renvs)
end
