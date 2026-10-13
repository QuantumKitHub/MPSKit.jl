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
include("overrides.jl")
include("src/assets/models.jl")

# examples — grouped by computational task; each group is a subdirectory of src/examples/
example_dir = joinpath(@__DIR__, "src", "examples")
example_groups = [
    "Ground states" => "groundstates",
    "Excitations & dispersions" => "excitations",
    "Dynamics & finite temperature" => "dynamics",
    "Statistical mechanics" => "statmech",
]
# the sidebar follows the reading order of the gallery index; examples it does not link come last
example_order = [
    m.captures[1] for m in eachmatch(
            r"\]\(([^)\s]+/index\.md)\)", read(joinpath(example_dir, "index.md"), String)
        )
]
example_pages = map(example_groups) do (title, group)
    pages = map(readdir(joinpath(example_dir, group))) do dir
        return joinpath(group, dir, "index.md")
    end
    sort!(pages; by = page -> something(findfirst(==(page), example_order), typemax(Int)))
    return title => map(page -> joinpath("examples", page), pages)
end

# contributing guide: `CONTRIBUTING.md` in the repository root is canonical, since that is the
# location GitHub links to from the issue and pull request forms. Copy it in as a page so it is
# reachable from the manual as well, with an `EditURL` pointing back at the real source file.
open(joinpath(@__DIR__, "src", "contributing.md"), "w") do io
    println(
        io, """
        ```@meta
        EditURL = "https://github.com/QuantumKitHub/MPSKit.jl/blob/main/CONTRIBUTING.md"
        ```
        """
    )
    return write(io, read(joinpath(@__DIR__, "..", "CONTRIBUTING.md"), String))
end

# bibliography
bibpath = joinpath(@__DIR__, "src", "assets", "mpskit.bib")
# TODO: Re-enable hover previews once the upstream fix is released:
# https://github.com/LuxDL/DocumenterVitepress.jl/pull/399
bib = CitationBibliography(bibpath; style = :authoryear, show_hover = false)

# interlinks
links = InterLinks(
    "TensorKit" => "https://quantumkithub.github.io/TensorKit.jl/stable/",
    "TensorOperations" => "https://quantumkithub.github.io/TensorOperations.jl/stable/",
    "KrylovKit" => "https://jutho.github.io/KrylovKit.jl/stable/",
    "BlockTensorKit" => "https://quantumkithub.github.io/BlockTensorKit.jl/dev/",
    "MatrixAlgebraKit" => "https://quantumkithub.github.io/MatrixAlgebraKit.jl/stable/"
)

# include MPSKit in all doctests
DocMeta.setdocmeta!(MPSKit, :DocTestSetup, :(using MPSKit, TensorKit); recursive = true)

makedocs(;
    sitename = "MPSKit.jl",
    format = DocumenterVitepress.MarkdownVitepress(;
        repo = "github.com/QuantumKitHub/MPSKit.jl",
        devbranch = "main",
        devurl = "dev",
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
        "Examples" => [
            "Overview" => "examples/index.md",
            example_pages...,
        ],
        "Library" => "lib/lib.md",
        "References" => "references.md",
        "Changelog" => "changelog.md",
        "Contributing" => "contributing.md",
    ],
    checkdocs = :exports,
    doctest = true,
    plugins = [bib, links, VitepressOverrides()]
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/QuantumKitHub/MPSKit.jl.git",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
