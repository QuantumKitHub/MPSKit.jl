# if examples is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

using MPSKit
using Documenter

example_dir = joinpath(@__DIR__, "src", "examples")
classic_pages = map(readdir(joinpath(example_dir, "classic2d"))) do dir
    return joinpath("examples", "classic2d", dir, "index.md")
end
quantum_pages = map(readdir(joinpath(example_dir, "quantum1d"))) do dir
    return joinpath("examples", "quantum1d", dir, "index.md")
end

# include MPSKit in all doctests
DocMeta.setdocmeta!(MPSKit, :DocTestSetup, :(using MPSKit); recursive=true)

mathengine = MathJax3(Dict(:loader => Dict("load" => ["[tex]/physics"]),
                           :tex => Dict("inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
                                        "tags" => "ams",
                                        "packages" => ["base", "ams", "autoload", "physics"])))
makedocs(;
         modules=[MPSKit],
         sitename="MPSKit.jl",
         format=Documenter.HTML(;
                                prettyurls=get(ENV, "CI", nothing) == "true",
                                mathengine,
                                size_threshold=512000),
         pages=["Home" => "index.md",
                "Manual" => ["man/intro.md",
                             "man/states.md",
                             "man/operators.md",
                             "man/algorithms.md",
                             "man/environments.md",
                             "man/parallelism.md"],
                "Examples" => "examples/index.md",
                "Library" => "lib/lib.md"],
         warnonly=true)

deploydocs(; repo="github.com/maartenvd/MPSKit.jl.git")
