struct AC2Spec{S <: ElementarySpace} # <: BenchmarkSpec
    physicalspaces::NTuple{2, S}
    mps_virtualspaces::NTuple{3, S}
    mpo_virtualspaces::NTuple{3, SumSpace{S}}
    nonzero_keys::NTuple{2, Vector{Tuple{Int, Int}}}
end

function AC2Spec(mps, mpo; site = length(mps) ÷ 2)
    physicalspaces = (physicalspace(mps, site), physicalspace(mps, site + 1))
    mps_virtualspaces = (left_virtualspace(mps, site), right_virtualspace(mps, site), right_virtualspace(mps, site + 1))
    mpo_virtualspaces = (left_virtualspace(mpo, site), right_virtualspace(mpo, site), right_virtualspace(mpo, site + 1))
    ks = (
        map(x -> (x.I[1], x.I[4]), nonzero_keys(mpo[site])),
        map(x -> (x.I[1], x.I[4]), nonzero_keys(mpo[site])),
    )
    return AC2Spec(physicalspaces, mps_virtualspaces, mpo_virtualspaces, ks)
end

# Benchmarks
# ----------
function MPSKit.MPO_AC2_Hamiltonian(spec::AC2Spec{S}; T::Type = Float64) where {S}
    GL = randn(T, spec.mps_virtualspaces[1] ⊗ spec.mpo_virtualspaces[1]' ← spec.mps_virtualspaces[1])
    GR = randn(T, spec.mps_virtualspaces[3] ⊗ spec.mpo_virtualspaces[3] ← spec.mps_virtualspaces[3])
    W1 = JordanMPOTensor{T, S}(undef, spec.mpo_virtualspaces[1] ⊗ spec.physicalspaces[1] ← spec.physicalspaces[1] ⊗ spec.mpo_virtualspaces[2])
    for (r, c) in spec.nonzero_keys[1]
        r == c == 1 && continue
        r == size(W1, 1) && c == size(W1, 4) && continue
        W1[r, 1, 1, c] = randn!(W1[r, 1, 1, c])
    end
    W2 = JordanMPOTensor{T, S}(undef, spec.mpo_virtualspaces[2] ⊗ spec.physicalspaces[2] ← spec.physicalspaces[2] ⊗ spec.mpo_virtualspaces[3])
    for (r, c) in spec.nonzero_keys[2]
        r == c == 1 && continue
        r == size(W2, 1) && c == size(W2, 4) && continue
        W2[r, 1, 1, c] = randn!(W2[r, 1, 1, c])
    end

    return MPSKit.MPO_AC2_Hamiltonian(GL, W1, W2, GR)
end

function contraction_benchmark(spec::AC2Spec; T::Type = Float64)
    AA = randn(T, spec.mps_virtualspaces[1] ⊗ spec.physicalspaces[1] ← spec.mps_virtualspaces[3] ⊗ spec.physicalspaces[2]')
    H_eff = MPSKit.MPO_AC2_Hamiltonian(spec; T)
    H_prep, x_prep = MPSKit.prepare_operator!!(H_eff, AA)
    init() = randn!(similar(x_prep))

    return @benchmarkable $H_prep * x setup = (x = $init())
end

function preparation_benchmark(spec::AC2Spec; T::Type = Float64)
    init() = randn(T, spec.mps_virtualspaces[1] ⊗ spec.physicalspaces[1] ← spec.mps_virtualspaces[3] ⊗ spec.physicalspaces[2]')
    H_eff = MPSKit.MPO_AC2_Hamiltonian(spec; T)

    return @benchmarkable begin
        O′, x′ = MPSKit.prepare_operator!!($H_eff, x)
        y = MPSKit.unprepare_operator!!(x′, O′, x)
    end setup = (x = $init())
end

# Converters
# ----------
function tomlify(spec::AC2Spec)
    return Dict(
        "physicalspaces" => collect(tomlify.(spec.physicalspaces)),
        "mps_virtualspaces" => collect(tomlify.(spec.mps_virtualspaces)),
        "mpo_virtualspaces" => collect(tomlify.(spec.mpo_virtualspaces)),
        "nonzero_keys" => collect(map(Base.Fix1(map, collect), spec.nonzero_keys))
    )
end

function untomlify(::Type{AC2Spec}, x)
    physicalspaces = Tuple(map(untomlify, x["physicalspaces"]))
    mps_virtualspaces = Tuple(map(untomlify, x["mps_virtualspaces"]))
    mpo_virtualspaces = Tuple(map(untomlify, x["mpo_virtualspaces"]))
    nonzero_keys = Tuple(map(Base.Fix1(map, Base.Fix1(map, Tuple)), x["nonzero_keys"]))
    return AC2Spec(physicalspaces, mps_virtualspaces, mpo_virtualspaces, nonzero_keys)
end

function Base.convert(::Type{AC2Spec{S₁}}, spec::AC2Spec{S₂}) where {S₁, S₂}
    return S₁ === S₂ ? spec : AC2Spec(
            S₁.(spec.physicalspaces),
            S₁.(spec.mps_virtualspaces),
            SumSpace{S₁}.(spec.mpo_virtualspaces),
            spec.nonzero_keys
        )
end
