# Home


## User manual
```@contents
Pages = ["man/intro.md","man/conventions.md","man/states.md","man/operators.md","man/algorithms.md","man/parallelism.md"]
Depth = 1
```

## Examples
```@contents
Pages = map(file -> joinpath("examples", file), 
            filter(f -> endswith(f, ".md"), readdir("examples")))
Depth = 1
```


## Library outline
```@contents
Pages = ["lib/lib.md"]
Depth = 1
```
