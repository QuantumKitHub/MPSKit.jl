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

    exci_space = space(exci[1][1],3);

    (numrows,numcols) = size(left_gs);

    st = site_type(typeof(left_gs));
    B_type = tensormaptype(spacetype(st),2,2,eltype(st));

    lBs = [PeriodicArray{B_type,1}(undef,size(left_gs,2)) for row in 1:size(left_gs,1)];
    rBs = [PeriodicArray{B_type,1}(undef,size(left_gs,2)) for row in 1:size(left_gs,1)];

    for row in 1:numrows

        c_lenvs = broadcast(col->leftenv(lenvs,col,left_gs)[row],1:numcols);
        c_renvs = broadcast(col->rightenv(renvs,col,right_gs)[row],1:numcols);

        hamrow = ham[row,:];

        left_above = left_gs[row];
        left_below = left_gs[row+1];
        right_above = right_gs[row];
        right_below = right_gs[row+1];

        left_renorms = fill(zero(eltype(B_type)),numcols);
        right_renorms = fill(zero(eltype(B_type)),numcols);

        for col in 1:numcols
            lv = leftenv(lenvs,col,left_gs)[row];
            rv = rightenv(lenvs,col,left_gs)[row];
            left_renorms[col] = @plansor lv[1 2;3]*left_above.AC[col][3 4;5]*hamrow[col][2 6;4 7]*rv[5 7;8]*conj(left_below.AC[col][1 6;8])

            lv = leftenv(renvs,col,right_gs)[row];
            rv = rightenv(renvs,col,right_gs)[row];
            right_renorms[col] = @plansor lv[1 2;3]*right_above.AC[col][3 4;5]*hamrow[col][2 6;4 7]*rv[5 7;8]*conj(right_below.AC[col][1 6;8])

        end

        left_renorms = left_renorms.^-1;
        right_renorms = right_renorms.^-1;

        lB_cur = TensorMap(zeros,eltype(B_type),
                                left_virtualspace(left_below,0)*_firstspace(hamrow[1])',
                                exci_space'*right_virtualspace(right_above,0));
        rB_cur = TensorMap(zeros,eltype(B_type),
                                left_virtualspace(left_below,0)*_firstspace(hamrow[1]),
                                exci_space'*right_virtualspace(right_above,0));
        for col in 1:numcols
            lB_cur = lB_cur*TransferMatrix(right_above.AR[col],hamrow[col],left_below.AL[col])
            lB_cur += c_lenvs[col]*TransferMatrix(exci[row][col],hamrow[col],left_below.AL[col])
            lB_cur *= left_renorms[col]*exp(-1im*exci.momentum);
            lBs[row][col] = lB_cur

            col = numcols-col+1;

            rB_cur = TransferMatrix(left_above.AL[col],hamrow[col],right_below.AR[col])*rB_cur
            rB_cur += TransferMatrix(exci[row][col],hamrow[col],right_below.AR[col])*c_renvs[col]
            rB_cur *= exp(1im*exci.momentum)*right_renorms[col]
            rBs[row][col] = rB_cur;
        end


        tm_RL = TransferMatrix(right_above.AR,hamrow,left_below.AL);
        tm_LR = TransferMatrix(left_above.AL,hamrow,right_below.AR);

        if exci.trivial
            @plansor rvec[-1 -2;-3] := rightenv(lenvs,0,left_gs)[row][-1 -2;1]*conj(left_below.CR[0][-3;1])
            @plansor lvec[-1 -2;-3] := leftenv(lenvs,1,left_gs)[row][-1 -2;1]*left_above.CR[0][1;-3]

            tm_RL = regularize(tm_RL,lvec,rvec);

            @plansor rvec[-1 -2;-3] := rightenv(renvs,0,right_gs)[row][1 -2;-3]*right_above.CR[0][-1;1]
            @plansor lvec[-1 -2;-3] := conj(right_below.CR[0][-3;1])*leftenv(renvs,1,right_gs)[row][-1 -2;1]

            tm_LR = regularize(tm_LR,lvec,rvec);
        end

        (lBs[row][end],convhist) = linsolve(flip(tm_RL),lB_cur,lB_cur,solver,1,-exp(-1im*numcols*exci.momentum)*prod(left_renorms))
        convhist.converged == 0 && @warn "lbe failed to converge $(convhist.normres)"

        (rBs[row][1],convhist) = linsolve(tm_LR,rB_cur,rB_cur,GMRES(),1,-exp(1im*numcols*exci.momentum)*prod(right_renorms))
        convhist.converged == 0 && @warn "rbe failed to converge $(convhist.normres)"


        left_cur = lBs[row][end];
        right_cur = rBs[row][1];
        for col in 1:numcols-1
            left_cur = left_renorms[col]*left_cur*TransferMatrix(right_above.AR[col],hamrow[col],left_below.AL[col])*exp(-1im*exci.momentum)
            lBs[row][col] += left_cur

            col = numcols-col+1
            right_cur = TransferMatrix(left_above.AL[col],hamrow[col],right_below.AR[col])*right_cur*exp(1im*exci.momentum)*right_renorms[col]
            rBs[row][col] += right_cur
        end
    end

    QPEnv(PeriodicArray(lBs),PeriodicArray(rBs),lenvs,renvs)
end
