import copy
import json
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


def test_compaction_keeps_corrections_and_whole_turns_without_editing_history(tmp_path: Path):
    agent = DefaultAgent(
        DeterministicModel(outputs=[]),
        LocalEnvironment(),
        system_template="system",
        instance_template="{{task}}",
        max_context_chars=850,
    )
    messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "original task"}]
    for i in range(12):
        messages += [
            make_output(f"command {i}: " + "x" * 100, [{"command": f"echo {i}"}]),
            {"role": "user", "content": f"output {i}: " + "y" * 100, "extra": {"returncode": 0}},
        ]
        if i in {2, 6}:
            messages.append(
                {"role": "user", "content": f"correction {i}", "extra": {"interrupt_type": "UserInterruption"}}
            )
    agent.messages = copy.deepcopy(messages)
    compacted = agent.compact(agent.messages)
    assert sum(len(m["content"]) for m in compacted) <= 850
    assert compacted[:2] == messages[:2]
    assert compacted[-2:] == messages[-2:]
    assert [m["content"] for m in compacted if m.get("extra", {}).get("interrupt_type")] == [
        "correction 2",
        "correction 6",
    ]
    assert "omitted" in compacted[2]["content"]
    for i, message in enumerate(compacted):
        if message.get("extra", {}).get("returncode") == 0:
            assert compacted[i - 1]["extra"]["actions"]
    assert agent.messages == messages
    path = tmp_path / "trajectory.json"
    agent.save(path)
    assert json.loads(path.read_text())["messages"] == messages


def test_essential_context_overflow_stops_before_sampling_and_saves(tmp_path: Path):
    agent = DefaultAgent(
        DeterministicModel(outputs=[]),
        LocalEnvironment(),
        system_template="system",
        instance_template="{{task}}",
        max_context_chars=100,
        output_path=tmp_path / "trajectory.json",
    )
    assert agent.run("task" * 100)["exit_status"] == "ContextLimitExceeded"
    saved = json.loads(agent.config.output_path.read_text())
    assert saved["info"]["model_stats"]["api_calls"] == 0
    assert saved["messages"][1]["content"] == "task" * 100
    assert "max_context_chars" in saved["messages"][-1]["content"]


def test_compaction_hook_controls_model_input_only(tmp_path: Path):
    class RecordingModel(DeterministicModel):
        def query(self, messages, **kwargs):
            self.received = messages
            return super().query(messages, **kwargs)

    class CustomAgent(DefaultAgent):
        def compact(self, messages):
            return [{"role": "user", "content": "custom compacted prompt"}]

    model = RecordingModel(outputs=[make_output("done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}])])
    agent = CustomAgent(
        model, LocalEnvironment(output_dir=tmp_path), system_template="system", instance_template="{{task}}"
    )
    assert agent.run("original task")["exit_status"] == "Submitted"
    assert model.received == [{"role": "user", "content": "custom compacted prompt"}]
    assert agent.messages[1]["content"] == "original task"
