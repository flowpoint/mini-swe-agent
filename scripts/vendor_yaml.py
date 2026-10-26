"""Refresh the pure-Python safe loader from an installed PyYAML 6.0.3."""

import ast
from importlib.metadata import distribution
from pathlib import Path

source = distribution("PyYAML")
if source.version != "6.0.3":
    raise SystemExit("Install PyYAML==6.0.3 before running this script")
target = Path(__file__).resolve().parents[1] / "src/minisweagent/_vendor/yaml"
target.mkdir(parents=True, exist_ok=True)

for name in ("error", "tokens", "events", "nodes", "reader", "scanner", "parser", "composer", "resolver"):
    (target / f"{name}.py").write_text(source.locate_file(f"yaml/{name}.py").read_text(encoding="utf-8"), encoding="utf-8")

for name, classes in (
    ("constructor", {"ConstructorError", "BaseConstructor", "SafeConstructor"}),
    ("loader", {"SafeLoader"}),
):
    content = source.locate_file(f"yaml/{name}.py").read_text(encoding="utf-8")
    nodes = [
        node
        for node in ast.parse(content).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or isinstance(node, ast.ClassDef)
        and node.name in classes
        or isinstance(node, ast.Expr)
        and ast.get_source_segment(content, node).startswith("SafeConstructor.")
    ]
    content = f"__all__ = {sorted(classes)!r}\n\n" + "\n\n".join(
        ast.get_source_segment(content, node) for node in nodes
    )
    (target / f"{name}.py").write_text(content.replace(", sys, types", ", types") + "\n", encoding="utf-8")

license_file = next(file for file in source.files if str(file).endswith("/licenses/LICENSE"))
(target / "LICENSE").write_text(source.locate_file(license_file).read_text(encoding="utf-8"), encoding="utf-8")
