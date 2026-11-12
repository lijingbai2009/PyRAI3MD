# Manual update helper

`tools/manual_update.py` compares the repository's `pyrai3md/` package
with the installed package and prints the `cp` commands that would bring
the install up to date, file by file. It never copies and never deletes:
a running calculation keeps the files it already loaded.

```bash
python tools/manual_update.py
python tools/manual_update.py --source /path/to/pyrai3md
```

## 1. What it compares

- Source: `Path(__file__).resolve().parents[1] / 'pyrai3md'` by default,
  or `--source`.
- Installed: the directory of the imported `pyrai3md` package.

Every regular file is compared with `filecmp.cmp(..., shallow=False)`
except `__pycache__` and `.pyc`/`.pyo`. The output has three sections:

| section | meaning |
|---|---|
| New files | present in the source, missing in the install |
| Changed files | present in both, content differs |
| Installed files with no source counterpart | present only in the install (never removed) |

An editable install points the installed package at the same tree; the
tool reports that and stops.

## 2. Why not reinstall

`pip install` replaces the whole package tree. A calculation that is
still running may import a module later, or a child process may read a
template; replacing files under it can break that run. Copying only the
changed files is the smaller, reversible action.

## 3. Applying the list

The output is a list of `cp source target` commands. Review it, then run
the commands you want, or copy a single file. Re-run the tool until the
new/changed sections are empty.

Only Python, YAML and data files that belong to the package are
compared; documentation under `dev/` and the `tools/` scripts
themselves are not part of the installed package.