import type { ShikiTransformer } from "shiki"

type PromptKind = "julia" | "pkg" | null

export function juliaReplTransformer(): ShikiTransformer {
  let promptInfoByLine: Array<{ len: number; kind: PromptKind }> = []
  let isJuliaBlock = false
  const rules: Array<{ kind: PromptKind; re: RegExp }> = [
    { kind: "julia", re: /^julia>/ },
    { kind: "pkg", re: /^(\([^)]*\)\s*)?pkg>/ },  // handles (@v1.9) pkg>
  ]

  function classify(line: string): { len: number; kind: PromptKind } {
    for (const r of rules) {
      const m = line.match(r.re)
      if (m) return { len: m[0].length, kind: r.kind }
    }

    return { len: 0, kind: null }
  }

  return {
    name: "julia-repl-prompts",

    preprocess(code, options) {
      isJuliaBlock = options.lang === "julia"
      return code
    },

    tokens(tokens) {
      if (!isJuliaBlock) {
        promptInfoByLine = []
        return
      }

      promptInfoByLine = tokens.map((lineTokens) => {
        const line = lineTokens.map((t) => t.content).join("")
        return classify(line)
      })
    },

    span(node, line, col) {
      if (!isJuliaBlock) return

      const info = promptInfoByLine[line - 1]
      if (!info || !info.kind || info.len <= 0) return

      if (col < info.len) {
        this.addClassToHast(node, "repl-prompt")
        this.addClassToHast(node, `repl-prompt-${info.kind}`)
      }
    },
  }
}
