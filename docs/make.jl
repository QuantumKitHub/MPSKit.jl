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

# examples — grouped by computational task; each group is a subdirectory of src/examples/
example_dir = joinpath(@__DIR__, "src", "examples")
example_groups = [
    "Ground states" => "groundstates",
    "Excitations & dispersions" => "excitations",
    "Dynamics & finite temperature" => "dynamics",
    "Statistical mechanics" => "statmech",
]
example_pages = map(example_groups) do (title, group)
    pages = map(readdir(joinpath(example_dir, group))) do dir
        return joinpath("examples", group, dir, "index.md")
    end
    return title => pages
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
            "tutorials/excitations.md",
            "tutorials/using_symmetries.md",
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
            "howto/parallelism_gpu.md",
        ],
        "Concepts" => [
            "concepts/vector_spaces.md",
            "concepts/matrix_product_states.md",
            "concepts/operators_and_hamiltonians.md",
            "concepts/symmetries.md",
            "concepts/algorithm_landscape.md",
            "concepts/environments.md",
            "concepts/parallelism_model.md",
        ],
        "Examples" => [
            "Overview" => "examples/index.md",
            example_pages...,
        ],
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
