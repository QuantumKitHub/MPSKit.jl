using Test

# check if user supplied args
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
const GROUP = if isnothing(arg_id)
    uppercase(get(ENV, "GROUP", "ALL"))
else
    uppercase(only(match(pat, ARGS[arg_id]).captures))
end

include("utilities/testsetup.jl")

@time begin
    if GROUP == "ALL" || GROUP == "STATES"
        @time include("states/finitemps.jl")
        @time include("states/infinitemps.jl")
        @time include("states/multilinemps.jl")
        @time include("states/windowmps.jl")
        @time include("states/quasiparticle.jl")
    end
    if GROUP == "ALL" || GROUP == "OPERATORS"
        @time include("operators/mpo.jl")
        @time include("operators/mpohamiltonian.jl")
        @time include("operators/lazysum.jl")
        @time include("operators/dense_mpo.jl")
        @time include("operators/projection.jl")
    end
    if GROUP == "ALL" || GROUP == "ALGORITHMS"
        @time include("algorithms/groundstate.jl")
        @time include("algorithms/timestep.jl")
        @time include("algorithms/excitations.jl")
        @time include("algorithms/changebonds.jl")
        @time include("algorithms/dynamical_dmrg.jl")
        @time include("algorithms/fidelity_susceptibility.jl")
        @time include("algorithms/correlators.jl")
        @time include("algorithms/approximate.jl")
        @time include("algorithms/periodic_boundary.jl")
        @time include("algorithms/taylorcluster.jl")
        @time include("algorithms/statmech.jl")
        @time include("algorithms/sector_conventions.jl")
    end
    if GROUP == "ALL" || GROUP == "MULTIFUSION"
        @time include("multifusion.jl")
    end
    if GROUP == "ALL" || GROUP == "OTHER"
        @time include("misc/aqua.jl")
        @time include("misc/plots.jl")
        @time include("misc/old_bugs.jl")
        @time include("misc/braille.jl")
        @time include("misc/styles.jl")
        @time include("misc/copy_behaviour.jl")
    end
end
