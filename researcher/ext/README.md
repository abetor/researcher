Extensions live in separate folders with their own README files. An extension may import the core, but the core must not import extensions; only CLI subcommand registration crosses that boundary.
