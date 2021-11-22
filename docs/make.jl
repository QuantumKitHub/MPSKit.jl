using Documenter, MPSKit

makedocs(modules=[MPSKit],
            sitename="MPSKit.jl",
            format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true",
                                        mathengine = MathJax()),
            pages = [
                "Home" => "index.md",
                "Manual" => ["man/intro.md", "man/conventions.md","man/states.md",
                                "man/operators.md", "man/algorithms.md","man/environments.md",
                                "man/parallelism.md"],
                "Tutorials" => ["tut/anyonic_statmech.md","tut/isingcft.md","tut/xxz_groundstate.md",
                                "tut/timeev.md","tut/haldane.md"],
                "Library" => ["lib/lib.md"],
            ])

deploydocs(
    repo = "github.com/maartenvd/MPSKit.jl.git",
)
