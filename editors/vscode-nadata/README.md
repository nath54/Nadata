# Nadata Language Support (VS Code / VSCodium)

Syntax highlighting for Nadata's two file formats:

- **`.nastruct`** — schema files (structs, attributes, methods, functions)
- **`.nadata`** — data files (instances, scripts, queries)

The grammar covers comments (line `#` and `"""` blocks), strings with
escapes, numbers, booleans, `self`, declaration keywords (`class`, `fn`,
`const`), control flow (`if` / `elif` / `for` / `while` / ...), logical
and compound operators, all primitive and generic types (`Int32`,
`UFloat16`, `Array[T]`, `Tensor[Float32, Batch, Dim]`, `GridSquare2D[T]`,
`DateTime`, `Color`, ...), class declarations with inheritance, and
`fn` / `@method` names.

## Install

### Packaged as a VSIX (recommended)

With Node.js available (no global install needed, `npx` fetches the
packager):

```bash
cd editors/vscode-nadata
npx -y @vscode/vsce package --allow-missing-repository
# produces nadata-support-<version>.vsix
code --install-extension nadata-support-<version>.vsix
codium --install-extension nadata-support-<version>.vsix
```

Generated `.vsix` files are gitignored; rebuild one any time with the
command above. Packaged releases attach the current `.vsix` to the
matching git tag.

### Manual copy

Copy (or symlink) the `editors/vscode-nadata` folder into the editor's
extensions directory:

- VS Code: `~/.vscode/extensions/vscode-nadata`
- VSCodium: `~/.vscodium/extensions/vscode-nadata`

Then reload the window.

## Keeping the grammar in sync

`tests/test_grammar.py` runs with the normal test suite and fails when:

- the grammar JSON is malformed or a scope name is missing,
- a builtin type name (extracted from the live
  `TypeResolver.builtin_types` registry plus the spatial grids and
  `Tensor`) is not highlighted by the grammar,
- the language keywords list drifts from the grammar.

When you add a language construct, update
`editors/vscode-nadata/syntaxes/nadata.tmLanguage.json` and the expected
lists in `tests/test_grammar.py` if needed — the test documents exactly
what is checked. The `.nastruct` grammar is a thin alias that includes
the whole `source.nadata` grammar, so both file types stay in sync
automatically.
