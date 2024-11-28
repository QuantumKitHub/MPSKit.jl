"""
Use krylovkit to perform exact diagonalization
"""
function exact_diagonalization(opp::MPOHamiltonian;
                               sector=first(sectors(oneunit(opp.pspaces[1]))),
                               len::Int=opp.period, num::Int=1, which::Symbol=:SR,
                               alg=Defaults.alg_eigsolve(; dynamic_tols=false))
    left = ℂ[typeof(sector)](sector => 1)
    right = oneunit(left)

    middle_site = Int(round(len / 2))

    Ot = eltype(opp[1])

    mpst_type = tensormaptype(spacetype(Ot), 2, 1, storagetype(Ot))
    mpsb_type = tensormaptype(spacetype(Ot), 1, 1, storagetype(Ot))
    CLs = Vector{Union{Missing,mpsb_type}}(missing, len + 1)
    ALs = Vector{Union{Missing,mpst_type}}(missing, len)
    ARs = Vector{Union{Missing,mpst_type}}(missing, len)
    ACs = Vector{Union{Missing,mpst_type}}(missing, len)

    for i in 1:(middle_site - 1)
        ALs[i] = isomorphism(storagetype(Ot), left * opp.pspaces[i],
                             fuse(left * opp.pspaces[i]))
        left = _lastspace(ALs[i])'
    end
    for i in len:-1:(middle_site + 1)
        ARs[i] = _transpose_front(isomorphism(storagetype(Ot),
                                              fuse(right * opp.pspaces[i]'),
                                              right * opp.pspaces[i]'))
        right = _firstspace(ARs[i])
    end
    ACs[middle_site] = randomize!(similar(opp[1][1, 1],
                                          left * opp.pspaces[middle_site] ← right))
    norm(ACs[middle_site]) == 0 && throw(ArgumentError("invalid sector"))
    normalize!(ACs[middle_site])

    #construct the largest possible finite mps of that length
    state = FiniteMPS(ALs, ARs, ACs, CLs)
    envs = environments(state, opp)

    #optimize the middle site. Because there is no truncation, this single site captures the entire possible hilbert space
    H_ac = ∂∂AC(middle_site, state, opp, envs) # this linear operator is now the actual full hamiltonian!
    (vals, vecs, convhist) = eigsolve(H_ac, state.AC[middle_site], num, which, alg)

    state_vecs = map(vecs) do v
        cs = copy(state)
        cs.AC[middle_site] = v
        return cs
    end

    return vals, state_vecs, convhist
end
