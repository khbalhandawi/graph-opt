# graph-opt

This is a library for graph algorithms in Python. 

## Getting started

To install this package use the `pdm` package manager. You can add this library to your `.pyproject.toml` file as 

```yaml
"graph-opt @ git+ssh://git@github.com:khbalhandawi/graph-opt.git@master",
```

and then simple use the 

```bash
pdm install
```

command to install all the required dependencies into a virtual environment

## Testing

Run the command to initiate all no GPU related tests use the command

```bash
pytest tests/test_community.py -k "not GPU" -v
```

## License

MIT; see [LICENSE](LICENSE).
