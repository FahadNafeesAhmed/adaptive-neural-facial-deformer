# Validate rig metadata

From the repository root in PowerShell, run:

```powershell
$env:PYTHONPATH = 'src'
python -m anf_deformer validate-rig 'path/to/rig.json'
```

On a POSIX shell:

```sh
PYTHONPATH=src python -m anf_deformer validate-rig path/to/rig.json
```

After installing the package, the same command is available as `anf-deformer validate-rig path/to/rig.json`.

The command reads UTF-8 metadata, applies the [rig metadata contract](../data/rig-metadata.md), and prints the number of controls and vertices. It does not modify the file or load a character asset.

Exit status is `0` for valid metadata, `1` for invalid or unreadable input, and `2` for incorrect command-line usage. Error messages go to standard error. Use `python -m anf_deformer --help` for command help.

Successful validation only establishes that the metadata satisfies the current contract. It does not verify a mesh, an exporter, or a trained model.
