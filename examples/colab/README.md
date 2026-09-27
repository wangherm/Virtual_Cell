# Archived Colab example

`VCell_TeacherStudent_Colab.ipynb` is the original **v0.2.1** teaching notebook,
preserved for reference. It expects the original v0.2.1 ZIP and is not the entry
point for the current server release. Its upload, Drive and source-push cells
describe the old release.

Use the [current README](../../README.md) and [server guide](../../docs/SERVER.md)
to clone, install, train and evaluate without notebook or Google services.
The historical Chinese walkthrough is in
[docs/legacy](../../docs/legacy/STRATEGY_AND_HOWTO_ZH.md).

To reproduce the original notebook, use the original supplied v0.2.1 ZIP, or
check out commit `8014f679a7f9a13975542f51bb481c14fcd431c4` in a separate directory
and run its `scripts/package_release.py`. Old notebook generation and token-based
source-push scripts remain available in that commit's history.

Optional notebook schema/syntax validation only:

```bash
python -m pip install -e '.[dev,notebook]'
python -m pytest -q optional_tests/test_notebook.py
```

This checks the archived document; it does not execute Colab cells or contact APIs.
