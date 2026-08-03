# Examples

This is a folder to generate the examples in the documentation. For now, this needs to be
updated manually.

In order to Trigger the file generation, run:

``julia examples/make.jl`

By default, this will only generate files when the input file has not changed. This is
achieved by keeping a checksum of the `main.jl` file in each example in `Cache.toml`.
Total recompilation can be achieved by deleting this file, or alternatively you can just
delete the entries for which you wish to generate new files.

## Contributing

Contributions are welcome! Please open an issue or a pull request if you have any questions
or suggestions. The code should be placed in a folder in one of the topic groups (`groundstates`,
`excitations`, `dynamics`, `statmech`), and the `main.jl` file should be the entry point.
A new topic group is picked up automatically: any subdirectory of `examples/` is built, so
only the sidebar grouping in `docs/make.jl` needs updating. Any other files will
be copied over to the `docs/src/examples` folder, so you can use this to include images or
other files.