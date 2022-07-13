"onesite tdvp"
@with_kw struct TDVP{A} <: Algorithm
    expalg::A = Lanczos(tol=Defaults.tol)
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
end

"""
    function timestep(psi, operator, dt, alg,envs = environments(psi,operator))

time evolves psi by timestep dt using algorithm alg
"""
function timestep(state::InfiniteMPS, H, timestep::Number,alg::TDVP,envs::Cache=environments(state,H))

    temp_ACs = similar(state.AC);
    temp_CRs = similar(state.CR);

    @sync for (loc,(ac,c)) in enumerate(zip(state.AC,state.CR))
        @Threads.spawn begin
            h = ∂∂AC($loc,$state,$H,$envs);
            ($temp_ACs[loc],convhist) = exponentiate(h ,-1im*$timestep,$ac,alg.expalg)
            convhist.converged==0 && @info "time evolving ac($loc) failed $(convhist.normres)"
        end

        @Threads.spawn begin
            h = ∂∂C($loc,$state,$H,$envs);
            ($temp_CRs[loc],convhist) = exponentiate(h, -1im*$timestep,$c,alg.expalg)
            convhist.converged==0 && @info "time evolving a($loc) failed $(convhist.normres)"
        end
    end

    for loc in 1:length(state)

        #find Al that best fits these new Acenter and centers
        QAc,_ = leftorth!(temp_ACs[loc],alg=TensorKit.QRpos())
        Qc,_ = leftorth!(temp_CRs[loc],alg=TensorKit.QRpos())
        @plansor temp_ACs[loc][-1 -2;-3] = QAc[-1 -2;1]*conj(Qc[-3;1])
    end

    nstate = InfiniteMPS(temp_ACs,state.CR[end]; tol = alg.tolgauge, maxiter = alg.maxiter)
    recalculate!(envs,nstate)
    nstate,envs
end

function timestep!(state::Union{FiniteMPS,MPSComoving}, H, timestep::Number,alg::TDVP,envs=environments(state,H))
    #left to right
    for i in 1:(length(state)-1)
        h_ac = ∂∂AC(i,state,H,envs);
        (state.AC[i],convhist) = exponentiate(h_ac,-1im*timestep/2,state.AC[i],alg.expalg)

        h_c = ∂∂C(i,state,H,envs);
        (state.CR[i],convhist) = exponentiate(h_c,1im*timestep/2,state.CR[i],alg.expalg)

    end

    h_ac = ∂∂AC(length(state),state,H,envs);
    (state.AC[end],convhist) = exponentiate(h_ac,-1im*timestep/2,state.AC[end],alg.expalg)

    #right to left
    for i in length(state):-1:2
        h_ac = ∂∂AC(i,state,H,envs);
        (state.AC[i],convhist) = exponentiate(h_ac,-1im*timestep/2,state.AC[i],alg.expalg)

        h_c = ∂∂C(i-1,state,H,envs);
        (state.CR[i-1],convhist) = exponentiate(h_c,1im*timestep/2,state.CR[i-1],alg.expalg)
    end

    h_ac = ∂∂AC(1,state,H,envs);
    (state.AC[1],convhist) = exponentiate(h_ac,-1im*timestep/2,state.AC[1],alg.expalg)

    return state,envs
end

"twosite tdvp (works for finite mps's)"
@with_kw struct TDVP2{A} <: Algorithm
    expalg::A = Lanczos(tol = Defaults.tol);
    tolgauge::Float64 = Defaults.tolgauge
    maxiter::Int = Defaults.maxiter
    trscheme = truncerr(1e-3)
end

#twosite tdvp for finite mps
function timestep!(state::Union{FiniteMPS,MPSComoving}, H, timestep::Number,alg::TDVP2,envs=environments(state,H);rightorthed=false)
    #left to right
    for i in 1:(length(state)-1)
        ac2 = _transpose_front(state.AC[i])*_transpose_tail(state.AR[i+1])

        h_ac2 = ∂∂AC2(i,state,H,envs);
        (nac2,convhist) = exponentiate(h_ac2,-1im*timestep/2,ac2,alg.expalg)

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme, alg=TensorKit.SVD())

        state.AC[i] = (nal,complex(nc))
        state.AC[i+1] = (complex(nc),_transpose_front(nar))

        if(i!=(length(state)-1))
            h_ac = ∂∂AC(i+1,state,H,envs);
            (state.AC[i+1],convhist) = exponentiate(h_ac,1im*timestep/2,state.AC[i+1],alg.expalg)
        end

    end

    #right to left
    for i in length(state):-1:2
        ac2 = _transpose_front(state.AL[i-1])*_transpose_tail(state.AC[i])

        h_ac2 = ∂∂AC2(i-1,state,H,envs);
        (nac2,convhist) = exponentiate(h_ac2,-1im*timestep/2,ac2,alg.expalg)

        (nal,nc,nar) = tsvd(nac2,trunc=alg.trscheme,alg=TensorKit.SVD())

        state.AC[i-1] = (nal,complex(nc))
        state.AC[i] = (complex(nc),_transpose_front(nar));

        if(i!=2)
            h_ac = ∂∂AC(i-1,state,H,envs);
            (state.AC[i-1],convhist) = exponentiate(h_ac,1im*timestep/2,state.AC[i-1],alg.expalg)
        end
    end

    return state,envs
end

#copying version
timestep(state::Union{FiniteMPS,MPSComoving},H,timestep,alg::Union{TDVP,TDVP2},envs=environments(state,H)) = timestep!(copy(state),H,timestep,alg,envs)
