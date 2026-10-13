# if examples is not the current active environment, switch to it
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path = (@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

using MPSKit
using Literate
using TOML, SHA

include(joinpath(@__DIR__, "figure_externalization.jl"))

# ---------------------------------------------------------------------------------------- #
# Caching
# ---------------------------------------------------------------------------------------- #

const CACHEFILE = joinpath(@__DIR__, "Cache.toml")
const MODELSFILE = joinpath(@__DIR__, "..", "docs", "src", "assets", "models.jl")
const MODEL_INCLUDE = "include(joinpath(@__DIR__, \"..\", \"..\", \"..\", \"docs\", \"src\", \"assets\", \"models.jl\")) #hide"

getcache() = isfile(CACHEFILE) ? TOML.parsefile(CACHEFILE) : Dict{String, Any}()

function iscached(root, name)
    cache = getcache()
    return haskey(cache, root) &&
        haskey(cache[root], name) &&
        cache[root][name] == checksum(root, name)
end

function setcached(root, name)
    cache = getcache()
    if haskey(cache, root)
        cache[root][name] = checksum(root, name)
    else
        cache[root] = Dict{String, Any}(name => checksum(root, name))
    end
    return open(f -> TOML.print(f, cache), CACHEFILE, "w")
end

# generate checksum based on path relative to ~/.../MPSKit.jl
# such that different users do not have to rerun already cached examples
function checksum(root, name)
    example_path = joinpath(@__DIR__, root, name, "main.jl")
    @assert isfile(example_path)
    inputs = (
        example_path, MODELSFILE, @__FILE__, joinpath(@__DIR__, "Project.toml"),
        joinpath(@__DIR__, "..", "Project.toml"),
    )
    return bytes2hex(sha256(mapreduce(read, vcat, inputs)))
end

# Inline the shared constructors so exported notebooks run without repository-relative paths.
# Hide their implementation in the rendered page; the source is available as a docs asset.
function inline_models(str; hide = false)
    contains(str, MODEL_INCLUDE) || return str
    source = replace(read(MODELSFILE, String), r"(?m)^(\h*)#" => s"\1##")
    if hide
        source = join((line * " #hide" for line in split(source, '\n')), '\n')
    end
    return replace(str, MODEL_INCLUDE => source)
end

# ---------------------------------------------------------------------------------------- #
# Building
# ---------------------------------------------------------------------------------------- #

attach_notebook_badge(root, name) = str -> attach_notebook_badge(root, name, str)
function attach_notebook_badge(root, name, str)
    mybinder_badge_url = "https://mybinder.org/badge_logo.svg"
    nbviewer_badge_url = "https://img.shields.io/badge/show-nbviewer-579ACA.svg"
    download_badge_url = "https://img.shields.io/badge/download-project-orange"
    mybinder = "[![]($mybinder_badge_url)](@__BINDER_ROOT_URL__/examples/$root/$name/main.ipynb)"
    nbviewer = "[![]($nbviewer_badge_url)](@__NBVIEWER_ROOT_URL__/examples/$root/$name/main.ipynb)"
    download = "[![]($download_badge_url)](https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/QuantumKitHub/MPSKit.jl/examples/tree/gh-pages/dev/examples/$root/$name)"

    markdown_only(x) = "#md # " * x
    return join(map(markdown_only, (mybinder, nbviewer, download)), "\n") * "\n\n" * str
end

# Log messages captured from an executed example carry the absolute source location of
# whatever emitted them, e.g.
#
#     └ @ MPSKit /home/someone/checkout/src/algorithms/groundstate/vumps.jl:87
#     └ @ OptimKit /home/someone/.julia/packages/OptimKit/K7Ujj/src/cg.jl:188
#
# Both depend on who ran the pipeline — the checkout path for a dev'ed package, and the
# depot slug for an installed one — so committing them makes the rendered pages differ
# per machine. The module name is already printed, so keep only the in-package path.
normalize_log_locations(content::AbstractString) =
    replace(content, r"(@ [A-Za-z_][A-Za-z0-9_]* )\S*?/(src/\S*\.jl:\d+)" => s"\1\2")

# A `using` block emits precompilation progress whenever the pipeline happens to run
# against a cold depot, e.g.
#
#     Precompiling packages...
#       14306.2 ms  ✓ MPSKitModels
#       1 dependency successfully precompiled in 16 seconds. 74 already precompiled.
#
# That says nothing about the example and its timings differ per machine, so drop any
# captured-output block whose every line is precompilation progress. Blocks that mix
# precompilation with real output are left alone.
function strip_precompilation_output(content::AbstractString)
    is_precompilation_line(line) = !isnothing(
        match(
            r"""^\s*(?:
                Precompiling\ .*                                  # the header
              | [\d.]+\s*ms\s*[✓✗].*                              # per-package timings
              | \d+\ dependenc(?:y|ies)\ successfully\ precompiled.*
              | \d+\ already\ precompiled\..*
            )?\s*$"""x, line
        )
    )

    lines = collect(eachsplit(content, '\n'))
    kept = similar(lines, 0)
    i = firstindex(lines)
    while i <= lastindex(lines)
        # Walk fenced blocks as blocks. Both ````julia (code) and bare ```` (captured
        # output) open one and a bare ```` closes it, so a closing fence must never be
        # mistaken for the start of the next block.
        if startswith(lines[i], "````")
            close = findnext(l -> rstrip(l) == "````", lines, i + 1)
            if !isnothing(close)
                body = @view lines[(i + 1):(close - 1)]
                if rstrip(lines[i]) == "````" &&
                        any(contains("Precompiling"), body) &&
                        all(is_precompilation_line, body)
                    # drop the block, and the blank line that followed it
                    i = close + 1
                    i <= lastindex(lines) && isempty(rstrip(lines[i])) && (i += 1)
                    continue
                end
                append!(kept, @view lines[i:close])
                i = close + 1
                continue
            end
        end
        push!(kept, lines[i])
        i += 1
    end
    return join(kept, '\n')
end

# All examples run in this one process, and Plots keeps `theme`/`default` settings as global
# state, so an example that styles its plots would otherwise restyle every later example.
const PLOTS_PKGID = Base.PkgId(Base.UUID("91a5bcdd-55d7-5caf-9e0b-520d859cae80"), "Plots")
function reset_plot_defaults()
    plots = get(Base.loaded_modules, PLOTS_PKGID, nothing)
    isnothing(plots) || Base.invokelatest(plots.reset_defaults)
    return nothing
end

function build_example(root, name)
    source_dir = joinpath(@__DIR__, "..", "examples", root, name)
    source_file = joinpath(source_dir, "main.jl")
    target_dir = joinpath(@__DIR__, "..", "docs", "src", "examples", root, name)

    return if !iscached(root, name)
        reset_plot_defaults()
        Literate.markdown(
            source_file, target_dir; execute = true, name = "index",
            preprocess = str -> attach_notebook_badge(root, name, inline_models(str; hide = true)),
            postprocess = content -> strip_precompilation_output(
                normalize_log_locations(externalize_figures(content, target_dir))
            ),
            mdstrings = true,
            nbviewer_root_url = "https://nbviewer.jupyter.org/github/QuantumKitHub/MPSKit.jl/blob/gh-pages/dev",
            binder_root_url = "https://mybinder.org/v2/gh/QuantumKitHub/MPSKit.jl/gh-pages?filepath=dev",
            credits = false,
            repo_root_url = "https://github.com/QuantumKitHub/MPSKit.jl"
        )
        Literate.notebook(
            source_file, target_dir; execute = false, name = "main",
            preprocess = str -> replace(inline_models(str), r"(?<!`)``(?!`)" => "\$"),
            mdstrings = true, credits = false
        )

        foreach(filter(!=("main.jl"), readdir(source_dir))) do f
            return cp(joinpath(source_dir, f), joinpath(target_dir, f); force = true)
        end
        setcached(root, name)
    end
end

function build(root)
    examples = readdir(joinpath(@__DIR__, root))
    return map(ex -> build_example(root, ex), examples)
end

# ---------------------------------------------------------------------------------------- #
# Scripts
# ---------------------------------------------------------------------------------------- #

# build every topic group: each subdirectory of examples/ is one group
for group in readdir(@__DIR__)
    startswith(group, '.') && continue
    isdir(joinpath(@__DIR__, group)) || continue
    build(group)
end
