using MPSKit
using Documenter
using Weave

example_in = abspath("examples")
example_out = abspath(joinpath("src", "examples"))

example_pages = []
for example in readdir(example_in)
    example == "cache" && continue
    if example == "assets"
        cp(joinpath(example_in, example), joinpath(example_out, example); force=true)
        continue
    end
    
    weave(joinpath(example_in, example);
        fig_ext=".svg", fig_path="figures",
        cache=:all, cache_path=joinpath(abspath(example_in), "cache"),
        out_path=example_out, doctype="github", keep_unicode=true)
    push!(example_pages, joinpath("examples", splitext(example)[1] * ".md"))
end

makedocs(; modules=[MPSKit],
         sitename="MPSKit.jl",
         format=Documenter.HTML(; 
              prettyurls=get(ENV, "CI", nothing) == "true",
              mathengine = MathJax3(Dict(
                     :loader => Dict("load" => ["[tex]/physics"]),
                     :tex => Dict(
                            "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
                            "tags" => "ams",
                            "packages" => ["base", "ams", "autoload", "physics"],
                     ),
                     ))),
         pages=["Home" => "index.md",
                "Manual" => ["man/intro.md", "man/conventions.md", "man/states.md",
                             "man/operators.md", "man/algorithms.md", "man/environments.md",
                             "man/parallelism.md"],
                "Examples" => example_pages,
                "Library" => ["lib/lib.md"]])

deploydocs(; repo="github.com/maartenvd/MPSKit.jl.git")
