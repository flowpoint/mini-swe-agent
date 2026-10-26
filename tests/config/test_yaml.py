import math

import pytest
import yaml as upstream

from minisweagent._vendor import yaml
from minisweagent.config import builtin_config_dir


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        ("# empty document\n", None),
        (
            "defaults: &defaults {temperature: 0.5, enabled: yes}\nmodel: {<<: *defaults, enabled: false}",
            {"defaults": {"temperature": 0.5, "enabled": True}, "model": {"temperature": 0.5, "enabled": False}},
        ),
        (
            'prompt: |\n  echo "héllo 世界"\n  printf \'%s\\n\' "$HOME"\nmodel: "01"\n',
            {"prompt": 'echo "héllo 世界"\nprintf \'%s\\n\' "$HOME"\n', "model": "01"},
        ),
        (
            "values: [null, true, false, 0x10, 1.5, .inf, -.inf]",
            {"values": [None, True, False, 16, 1.5, float("inf"), float("-inf")]},
        ),
    ],
)
def test_safe_loader_preserves_yaml_semantics(document, expected):
    assert yaml.safe_load(document) == upstream.safe_load(document) == expected


@pytest.mark.parametrize(
    ("document", "error"),
    [
        ("model: [unterminated", "ParserError"),
        ("model: *missing", "ComposerError"),
        ("---\nfirst\n---\nsecond\n", "ComposerError"),
        ("!!python/object/apply:builtins.len [[1, 2]]", "ConstructorError"),
        ("!!python/name:os.system", "ConstructorError"),
    ],
)
def test_safe_loader_rejects_invalid_documents_and_python_objects(document, error):
    with pytest.raises(yaml.YAMLError) as caught:
        yaml.safe_load(document)
    assert type(caught.value).__name__ == error


def test_safe_loader_preserves_aliases_and_bundled_configs():
    data = yaml.safe_load("items: &items [one, two]\ncopy: *items\nrecursive: &recursive [*recursive]\nnan: .nan")
    assert data["items"] is data["copy"]
    assert data["recursive"][0] is data["recursive"]
    assert math.isnan(data["nan"])
    for path in builtin_config_dir.glob("*.yaml"):
        assert yaml.safe_load(path.read_text()) == upstream.safe_load(path.read_text())
