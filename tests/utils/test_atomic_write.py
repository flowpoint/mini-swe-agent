import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel
from minisweagent.utils.serialize import atomic_write


def test_atomic_replacements_preserve_links_and_never_expose_partial_json(tmp_path: Path):
    path, link = tmp_path / "run.json", tmp_path / "latest.json"
    atomic_write(path, json.dumps({"value": "x" * 100000}))
    link.symlink_to(path.name)

    def write_many():
        for i in range(30):
            atomic_write(link, json.dumps({"value": str(i) * 100000}))

    with ThreadPoolExecutor() as executor:
        future = executor.submit(write_many)
        while not future.done():
            assert len(json.loads(link.read_text())["value"]) >= 100000
        future.result()
    assert link.is_symlink()
    assert json.loads(path.read_text())["value"] == "29" * 100000
    with pytest.raises(FileExistsError):
        atomic_write(path, "replacement", overwrite=False)
    assert json.loads(path.read_text())["value"] == "29" * 100000
    assert sorted(p.name for p in tmp_path.iterdir()) == ["latest.json", "run.json"]


def test_serialization_failure_preserves_previous_trajectory(tmp_path: Path):
    agent = DefaultAgent(DeterministicModel(outputs=[]), LocalEnvironment(), system_template="s", instance_template="t")
    path = tmp_path / "trajectory.json"
    agent.save(path)
    previous = path.read_bytes()
    agent.messages.append({"role": "user", "content": "oops", "extra": {"unsupported": object()}})
    with pytest.raises(TypeError):
        agent.save(path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]
