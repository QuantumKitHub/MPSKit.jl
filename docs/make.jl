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
using SHA: sha256

# `src/.vitepress/config.mts` is vendored from the DocumenterVitepress template (see the
# header comment there) so that we can hook `themeConfig.search` and `markdown.config`,
# which `MarkdownVitepress` does not expose. Warn when upstream changes the template, so
# that our copy can be re-synced — or dropped, once the fixes land upstream.
let template = joinpath(pkgdir(DocumenterVitepress), "template", "src", ".vitepress", "config.mts")
    # Templates the vendored copy is known to be a faithful superset of; v0.3.4 and v0.3.5
    # differ only by the (inert for us) NOINDEX marker.
    vendored_from = (
        "ca5a958eb398b3219557633f017467cfa07f4882dc2dfe55b12d1f6c0e70d729", # v0.3.4
        "56289223983a3844417eae597f81da878f395792a529721e7873178c92d60721", # v0.3.5
    )
    actual = bytes2hex(sha256(read(template)))
    actual in vendored_from || @warn """
    DocumenterVitepress' `config.mts` template has changed since `docs/src/.vitepress/config.mts` \
    was vendored from it. Re-sync the vendored copy (keeping the `MPSKit:` additions), or delete \
    it if upstream now ships them.""" template actual
end

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
            "howto/statmech.md",
            "howto/quasi_1d_geometries.md",
            "howto/convergence_troubleshooting.md",
            "howto/parallelism_gpu.md",
            "howto/saving_loading.md",
        ],
        "Concepts" => [
            "concepts/vector_spaces.md",
            "concepts/matrix_product_states.md",
            "concepts/finite_vs_infinite.md",
            "concepts/operators_and_hamiltonians.md",
            "concepts/symmetries.md",
            "concepts/algorithm_landscape.md",
            "concepts/environments.md",
            "concepts/parallelism_model.md",
            "concepts/numerics.md",
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
            "lib/environments.md",
            "lib/internals.md",
            "lib/lib.md",
        ],
        "References" => "references.md",
        "Changelog" => "changelog.md",
        "Migration" => "migration.md",
        "Contributing" => "contributing.md",
        "Citing" => "citing.md",
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
