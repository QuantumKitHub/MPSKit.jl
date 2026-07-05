# if examples is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.resolve()
    Pkg.instantiate()
end

using MPSKit
using Documenter
using DocumenterVitepress
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
    "TensorKit" => "https://quantumkithub.github.io/TensorKit.jl/stable/",
    "TensorOperations" => "https://quantumkithub.github.io/TensorOperations.jl/stable/",
    "KrylovKit" => "https://jutho.github.io/KrylovKit.jl/stable/",
    "BlockTensorKit" => "https://quantumkithub.github.io/BlockTensorKit.jl/dev/",
    "MatrixAlgebraKit" => "https://quantumkithub.github.io/MatrixAlgebraKit.jl/stable/",
    "MPSKitModels" => "https://quantumkithub.github.io/MPSKitModels.jl/dev/"
)

# include MPSKit in all doctests
DocMeta.setdocmeta!(MPSKit, :DocTestSetup, :(using MPSKit, TensorKit); recursive = true)

# root CHANGELOG.md is canonical (visible on GitHub); copy it in for rendering
cp(joinpath(@__DIR__, "..", "CHANGELOG.md"), joinpath(@__DIR__, "src", "changelog.md"); force = true)

makedocs(;
    sitename = "MPSKit.jl",
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/QuantumKitHub/MPSKit.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "tutorials/installation.md",
            "tutorials/first_groundstate.md",
            "tutorials/thermodynamic_limit.md",
            "tutorials/time_evolution.md",
        ],
        "Manual" => [
            "man/intro.md",
            "man/states.md",
            "man/operators.md",
            "man/algorithms.md",
            # "man/environments.md",
            "man/parallelism.md",
        ],
        "How-to" => [
            "howto/index.md",
            "howto/states.md",
            "howto/hamiltonians.md",
            "howto/groundstate_algorithms.md",
            "howto/bond_dimension.md",
            "howto/time_evolution.md",
            "howto/observables.md",
            "howto/entanglement.md",
            "howto/excitations.md",
        ],
        "Concepts" => [
            "concepts/algorithm_landscape.md",
        ],
        "Examples" => "examples/index.md",
        "Library" => [
            "lib/public.md",
            "lib/states.md",
            "lib/operators.md",
            "lib/groundstate.md",
            "lib/bond_dimension.md",
            "lib/time_evolution.md",
            "lib/excitations.md",
            "lib/observables.md",
            "lib/lib.md",
        ],
        "References" => "references.md",
        "Changelog" => "changelog.md",
    ],
    checkdocs = :exports,
    doctest = true,
    plugins = [bib, links]
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/QuantumKitHub/MPSKit.jl.git",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
