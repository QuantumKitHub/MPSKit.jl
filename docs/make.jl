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

# examples
example_dir = joinpath(@__DIR__, "src", "examples")
classic_pages = map(readdir(joinpath(example_dir, "classic2d"))) do dir
    return joinpath("examples", "classic2d", dir, "index.md")
end
quantum_pages = map(readdir(joinpath(example_dir, "quantum1d"))) do dir
    return joinpath("examples", "quantum1d", dir, "index.md")
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
        "Examples" => "examples/index.md",
        "Library" => "lib/lib.md",
        "References" => "references.md",
        "Changelog" => "changelog.md",
        "Contributing" => "contributing.md",
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
