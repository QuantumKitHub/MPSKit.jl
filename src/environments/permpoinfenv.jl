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

environments(state::InfiniteMPS,opp::DenseMPO;kwargs...) = environments(convert(MPSMultiline,state),convert(MPOMultiline,opp);kwargs...);
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

# --- utility functions ---

function gen_init_fps(above::MPSMultiline,mpo::Multiline{<:DenseMPO},below::MPSMultiline)
    T = eltype(above)

    map(1:size(mpo,1)) do cr
        L0::T = TensorMap(rand,eltype(T),space(below.AL[cr,1],1)*space(mpo[cr,1],1)',space(above.AL[cr,1],1))
        R0::T = TensorMap(rand,eltype(T),space(above.AR[cr,1],1)*space(mpo[cr,1],1),space(below.AR[cr,1],1))
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
            lw[j] = TensorMap(rand,eltype(A),_firstspace(be.AL[1])*ham[1].domspaces[j]',_firstspace(ab.AL[1]))
            rw[j] = TensorMap(rand,eltype(A),_lastspace(ab.AR[end])'*ham[end].imspaces[j]',_lastspace(be.AR[end])')
        end

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

    @sync for cr = 1:numrows
        @Threads.spawn begin
            (L0,R0) = $init[cr]

            shouldpack = L0 isa Vector;
            @sync begin
                @Threads.spawn begin
                    E_LL = TransferMatrix($above.AL[cr,:],$mpo[cr,:],$below.AL[cr+1,:])
                    (_,Ls,convhist) = eigsolve(shouldpack ? RecursiveVec($L0) : $L0,1,:LM,$solver) do x
                        y = E_LL*(shouldpack ? x.vecs : x)
                        shouldpack ? RecursiveVec(y) : y
                    end
                    convhist.converged < 1 && @info "left eigenvalue failed to converge $(convhist.normres)"
                    L0 = shouldpack ? Ls[1][:] : Ls[1];
                end
                @Threads.spawn begin
                    E_RR = TransferMatrix($above.AR[cr,:],$mpo[cr,:],$below.AR[cr+1,:])
                    (_,Rs,convhist) = eigsolve(shouldpack ? RecursiveVec($R0) : $R0,1,:LM,$solver) do x
                        y = E_RR*(shouldpack ? x.vecs : x)
                        shouldpack ? RecursiveVec(y) : y
                    end
                    convhist.converged < 1 && @info "right eigenvalue failed to converge $(convhist.normres)"
                    R0 = shouldpack ? Rs[1][:] : Rs[1];
                end
            end


            $lefties[cr,1] = L0;
            for loc in 2:numcols
                $lefties[cr,loc] = $lefties[cr,loc-1]*TransferMatrix($above.AL[cr,loc-1],$mpo[cr,loc-1],$below.AL[cr+1,loc-1])
            end


            renormfact::eltype(T) = dot($below.CR[cr+1,0],C_eff(L0,R0)*$above.CR[cr,0])

            $righties[cr,end] = R0/sqrt(renormfact);
            $lefties[cr,1] /=sqrt(renormfact);

            for loc in numcols-1:-1:1
                $righties[cr,loc] = TransferMatrix($above.AR[cr,loc+1],$mpo[cr,loc+1],$below.AR[cr+1,loc+1])*$righties[cr,loc+1]

                renormfact = dot($below.CR[cr+1,loc],C_eff($lefties[cr,loc+1],$righties[cr,loc])*$above.CR[cr,loc])
                $righties[cr,loc]/=sqrt(renormfact)
                $lefties[cr,loc+1]/=sqrt(renormfact)
            end
        end
    end

    return (lefties,righties)
end
