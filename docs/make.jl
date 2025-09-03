# if examples is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path = (@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

using MPSKit
using Documenter
using DocumenterCitations
using DocumenterInterLinks

# examples
example_dir = joinpath(@__DIR__, "src", "examples")
classic_pages = map(readdir(joinpath(example_dir, "classic2d"))) do dir
    return joinpath("examples", "classic2d", dir, "index.md")
end
quantum_pages = map(readdir(joinpath(example_dir, "quantum1d"))) do dir
    return joinpath("examples", "quantum1d", dir, "index.md")
end

# bibliography
bibpath = joinpath(@__DIR__, "src", "assets", "mpskit.bib")
bib = CitationBibliography(bibpath; style = :authoryear)

# interlinks
links = InterLinks(
    "TensorKit" => "https://jutho.github.io/TensorKit.jl/stable/",
    "TensorOperations" => "https://quantumkithub.github.io/TensorOperations.jl/stable/",
    "KrylovKit" => "https://jutho.github.io/KrylovKit.jl/stable/",
    "BlockTensorKit" => "https://lkdvos.github.io/BlockTensorKit.jl/dev/"
)

# include MPSKit in all doctests
DocMeta.setdocmeta!(MPSKit, :DocTestSetup, :(using MPSKit, TensorKit); recursive = true)

mathengine = MathJax3(
    Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
            "tags" => "ams",
            "packages" => ["base", "ams", "autoload", "physics"]
        )
    )
)
makedocs(;
    sitename = "MPSKit.jl",
    format = Documenter.HTML(;
        prettyurls = true,
        mathengine,
        assets = ["assets/custom.css"],
        size_threshold = 1024000,
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "man/intro.md",
            "man/states.md",
            "man/operators.md",
            "man/algorithms.md",
            # "man/environments.md",
            "man/parallelism.md",
            "man/lattices.md",
        ],
        "Examples" => "examples/index.md",
        "Library" => "lib/lib.md",
        "References" => "references.md",
    ],
    checkdocs = :exports,
    doctest = true,
    plugins = [bib, links]
)

deploydocs(; repo = "github.com/QuantumKitHub/MPSKit.jl.git", push_preview = true)
