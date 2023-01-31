# --- above === below ---
"
    This object manages the periodic mpo environments for an MPSMultiline
"
mutable struct PerMPOInfEnv{H,V,S<:MPSMultiline,A} <: AbstractInfEnv
    above :: Union{S,Nothing}

    opp :: H

    dependency :: S
    solver::A

    lw :: PeriodicArray{V,2}
    rw :: PeriodicArray{V,2}

    lock :: ReentrantLock
end

environments(state::InfiniteMPS,opp::DenseMPO;kwargs...) =
    environments(convert(MPSMultiline,state),convert(MPOMultiline,opp);kwargs...);

function environments(state::MPSMultiline,mpo::MPOMultiline;solver=Defaults.eigsolver)
    (lw,rw) = mixed_fixpoints(state,mpo,state;solver)

    PerMPOInfEnv(nothing,mpo,state,solver,lw,rw,ReentrantLock())
end

function environments(below::InfiniteMPS,toapprox::Tuple{<:Union{SparseMPO,DenseMPO},<:InfiniteMPS};kwargs...)
    (opp,above) = toapprox
    environments(convert(MPSMultiline,below),(convert(MPOMultiline,opp),convert(MPSMultiline,above));kwargs...);
end
function environments(below::MPSMultiline,toapprox::Tuple{<:MPOMultiline,<:MPSMultiline};solver = Defaults.eigsolver)
    (mpo,above) = toapprox;
    (lw,rw) = mixed_fixpoints(above,mpo,below;solver)

    PerMPOInfEnv(above,mpo,below,solver,lw,rw,ReentrantLock())
end



recalculate!(envs::PerMPOInfEnv,nstate::InfiniteMPS) = recalculate!(envs,convert(MPSMultiline,nstate));
function recalculate!(envs::PerMPOInfEnv,nstate::MPSMultiline)
    sameDspace = reduce(&,_firstspace.(envs.dependency.CR) .== _firstspace.(nstate.CR))


    above = isnothing(envs.above) ? nstate : envs.above;
    init = collect(zip(envs.lw[:,1],envs.rw[:,end]))
    if !sameDspace
        init = gen_init_fps(above,envs.opp,nstate)
    end

    (envs.lw,envs.rw) = mixed_fixpoints(above,envs.opp,nstate,init,solver = envs.solver);
    envs.dependency = nstate;

    envs
end

function leftenv(envs::PerMPOInfEnv,pos::Int,state::InfiniteMPS)
    check_recalculate!(envs,state);
    envs.lw[1,pos]
end

function rightenv(envs::PerMPOInfEnv,pos::Int,state::InfiniteMPS)
    check_recalculate!(envs,state);
    envs.rw[1,pos]
end

function leftenv(envs::PerMPOInfEnv,pos::Int,state::MPSMultiline)
    check_recalculate!(envs,state);
    envs.lw[:,pos]
end

function rightenv(envs::PerMPOInfEnv,pos::Int,state::MPSMultiline)
    check_recalculate!(envs,state);
    envs.rw[:,pos]
end

function leftenv(envs::PerMPOInfEnv,row::Int,col::Int,state)
    check_recalculate!(envs,state);
    envs.lw[row,col]
end

function rightenv(envs::PerMPOInfEnv,row::Int,col::Int,state)
    check_recalculate!(envs,state);
    envs.rw[row,col]
end


# --- utility functions ---

function gen_init_fps(above::MPSMultiline,mpo::Multiline{<:DenseMPO},below::MPSMultiline)
    T = eltype(above)

    map(1:size(mpo,1)) do cr
        L0::T = randomize!(similar(above.AL[1,1],left_virtualspace(below,cr+1,0)*_firstspace(mpo[cr,1])',left_virtualspace(above,cr,0)))
        R0::T = randomize!(similar(above.AL[1,1],right_virtualspace(above,cr,0)*_firstspace(mpo[cr,1]),right_virtualspace(below,cr+1,0)))
        (L0,R0)
    end
end

function gen_init_fps(above::MPSMultiline,mpo::Multiline{<:SparseMPO},below::MPSMultiline)
    map(1:size(mpo,1)) do cr
        ham = mpo[cr];
        ab = above[cr];
        be = below[cr];

        A = eltype(ab);

        lw = Vector{A}(undef,ham.odim)
        rw = Vector{A}(undef,ham.odim)

        for j = 1:ham.odim
            lw[j] = similar(ab.AL[1],_firstspace(be.AL[1])*ham[1].domspaces[j]',_firstspace(ab.AL[1]))
            rw[j] = similar(ab.AL[1],_lastspace(ab.AR[end])'*ham[end].imspaces[j]',_lastspace(be.AR[end])')
        end

        randomize!.(lw);
        randomize!.(rw);

        (lw,rw)
    end
end

function mixed_fixpoints(above::MPSMultiline,mpo::MPOMultiline,below::MPSMultiline,init = gen_init_fps(above,mpo,below);solver = Defaults.eigsolver)
    T = eltype(above);

    #sanity check
    (numrows,numcols) = size(above)
    @assert size(above) == size(mpo)
    @assert size(below) == size(mpo);

    envtype = eltype(init[1]);
    lefties = PeriodicArray{envtype,2}(undef,numrows,numcols);
    righties = PeriodicArray{envtype,2}(undef,numrows,numcols);

    @threads for cr = 1:numrows
        c_above = above[cr];
        c_below = below[cr+1];


        (L0,R0) = init[cr]

        @sync begin
            @Threads.spawn begin
                E_LL = TransferMatrix($c_above.AL,$mpo[cr,:],$c_below.AL)

                packed_init = $L0 isa Vector ? RecursiveVec($L0) : $L0;
                (_,Ls,convhist) = eigsolve(flip(E_LL),packed_init,1,:LM,$solver)
                convhist.converged < 1 && @info "left eigenvalue failed to converge $(convhist.normres)"
                L0 = $L0 isa Vector ? Ls[1].vecs : Ls[1];
            end
            @Threads.spawn begin

                packed_init = $R0 isa Vector ? RecursiveVec($R0) : $R0;
                E_RR = TransferMatrix($c_above.AR,$mpo[cr,:],$c_below.AR)
                (_,Rs,convhist) = eigsolve(E_RR, packed_init,1,:LM,$solver)
                convhist.converged < 1 && @info "right eigenvalue failed to converge $(convhist.normres)"
                R0 = $R0 isa Vector ? Rs[1].vecs : Rs[1];
            end
        end

        lefties[cr,1] = L0;
        for loc in 2:numcols
            lefties[cr,loc] = lefties[cr,loc-1]*TransferMatrix(c_above.AL[loc-1],mpo[cr,loc-1],c_below.AL[loc-1])
        end

        renormfact::eltype(T) = dot(c_below.CR[0],MPO_∂∂C(L0,R0)*c_above.CR[0])

        righties[cr,end] = R0/sqrt(renormfact);
        lefties[cr,1] /=sqrt(renormfact);

        for loc in numcols-1:-1:1
            righties[cr,loc] = TransferMatrix(c_above.AR[loc+1],mpo[cr,loc+1],c_below.AR[loc+1])*righties[cr,loc+1]

            renormfact = dot(c_below.CR[loc],MPO_∂∂C(lefties[cr,loc+1],righties[cr,loc])*c_above.CR[loc])
            righties[cr,loc]/=sqrt(renormfact)
            lefties[cr,loc+1]/=sqrt(renormfact)
        end
    end

    return (lefties,righties)
end
