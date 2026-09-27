"""Optional validation of the archived v0.2.1 notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_schema_and_code_syntax():
    import ast
    import nbformat
    nb = nbformat.read(ROOT / "examples/colab/VCell_TeacherStudent_Colab.ipynb", as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None
            assert not cell.outputs
