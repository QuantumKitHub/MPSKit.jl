# Centralized figure externalization for the documentation examples.
#
# `Literate.markdown(...; execute = true)` embeds figures produced by the examples as
# base64 data URIs inside `@raw html` blocks, e.g.
#
#     ```@raw html
#     <img src="data:image/png;base64,iVBOR...."/>
#     ```
#
# which bloats the generated `index.md` files. The helpers here rewrite those blocks to
# standard markdown image references (`![](figure-N.png)`) and write the decoded image bytes
# to external files next to the page. File-referencing `<img>`/`![]` tags (e.g. the committed
# `*.svg` diagrams) are left untouched — only `data:...;base64,` sources are externalized.

using Base64

# A `@raw html` fenced block whose body is a single `<img ...>` tag.
const _RAW_IMG_BLOCK = r"```@raw html[ \t]*\r?\n[ \t]*(<img[^>]*>)[ \t]*\r?\n```"
# A base64 image data URI inside such a tag: capture the MIME subtype and the payload.
const _DATA_URI = r"data:image/([A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)"

_ext_for(fmt) = fmt == "svg+xml" ? "svg" : fmt

"""
    externalize_figures(content::AbstractString, dir; prefix="figure") -> String

Rewrite base64 `@raw html` image blocks in `content` to `![](<prefix>-N.<ext>)`, writing the
decoded bytes to `<dir>/<prefix>-N.<ext>`. Blocks without a `data:...;base64,` source are left
unchanged. Returns the rewritten markdown.
"""
function externalize_figures(content::AbstractString, dir::AbstractString; prefix = "figure")
    n = 0
    return replace(
        content, _RAW_IMG_BLOCK => function (block)
            m = match(_DATA_URI, block)
            m === nothing && return block   # e.g. an <img src="./foo.svg"> — leave it alone
            ext = _ext_for(m.captures[1])
            n += 1
            fname = "$(prefix)-$(n).$(ext)"
            mkpath(dir)
            write(joinpath(dir, fname), base64decode(m.captures[2]))
            return "![]($(fname))"
        end
    )
end

"""
    externalize_figures!(index_md_path; prefix="figure") -> Int

Apply [`externalize_figures`](@ref) to the file at `index_md_path` in place, writing the image
files into the page's own directory. Returns the number of figures externalized (0 leaves the
file untouched). Idempotent.
"""
function externalize_figures!(index_md_path::AbstractString; prefix = "figure")
    content = read(index_md_path, String)
    nfigs = length(collect(eachmatch(_DATA_URI, content)))
    new = externalize_figures(content, dirname(index_md_path); prefix)
    new != content && write(index_md_path, new)
    return nfigs
end
