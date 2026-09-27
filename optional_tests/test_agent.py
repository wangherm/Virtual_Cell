import json
from pathlib import Path
import httpx
import numpy as np
import pytest
from openai import OpenAI
from vcell.agent import Decision, LLMPlanner, Settings, observation, run_agent, validate_decision
from vcell.losses import student_losses
from vcell.train import run_suite, settings_for
from vcell.synthetic import make_demo
from vcell.utils import read_config

ROOT = Path(__file__).resolve().parents[1]


def decision(settings, **changes):
    return Decision(action="train", assessment="测试用固定决策，不是真实 LLM 优化结果",
                    rationale="验证执行链路", settings=Settings(**{**settings, **changes}))


@pytest.fixture
def run(tmp_path):
    data = make_demo(tmp_path / "data", genes=24, targets=8, cells=8)
    cfg = read_config(ROOT / "configs/smoke.yaml")
    cfg.update(data_dir=str(data), output_dir=str(tmp_path / "run"), device="cpu",
               seeds=[0, 1], teacher_epochs=2, student_epochs=2, bootstrap_repeats=20)
    run_suite(cfg)
    return Path(cfg["output_dir"]), cfg


def test_agent_trials_budget_selection_and_resume(run):
    path, cfg = run
    teacher_files = list(path.glob("seed_*/teachers/*/best.pt"))
    before = {p: p.stat().st_mtime_ns for p in teacher_files}
    calls = []
    def planner(obs):
        calls.append(obs)
        return decision(settings_for(cfg, "mutual_contrastive"), learning_rate=.0008,
                        teacher_weights=[.5, .5, 0, 0])
    result = run_agent(path, "test-only-scripted-planner", max_trials=2, planner=planner)
    assert len(calls) == result["trials_completed"] == 2
    assert calls[1]["remaining_trials"] == 1
    assert len(calls[1]["trials"]) == 6
    assert before == {p: p.stat().st_mtime_ns for p in teacher_files}
    final = json.loads((path / "agent/final_observation.json").read_text())
    expected = min(final["trials"], key=lambda t: t["score"])
    assert result["selected_trial"] == expected["name"]
    assert result["validation_mse"] == expected["score"]
    assert result["student_model_epoch_cap"] == 2 * 4 * 2 * 2
    assert not (path / "evaluation_with_test").exists()
    for seed in cfg["seeds"]:
        with np.load(path / f"seed_{seed}/predictions.npz") as p:
            assert np.array_equal(p["agent_selected"], p[f"{expected['name']}/valmix"])
    def no_more_calls(obs):
        pytest.fail("Finished resume must not call API")
    assert run_agent(path, "test-only-scripted-planner", 2, True, no_more_calls) == result


def test_agent_pending_trial_resumes_without_new_decision(run, monkeypatch):
    import vcell.agent as module
    path, cfg = run
    calls = []
    def planner(obs):
        calls.append(obs)
        return decision(settings_for(cfg, "kd"))
    actual = module.run_trial
    def interrupted(*args, **kwargs):
        raise RuntimeError("Colab interrupted")
    monkeypatch.setattr(module, "run_trial", interrupted)
    with pytest.raises(RuntimeError, match="Colab"):
        run_agent(path, "test-only-scripted-planner", 1, planner=planner)
    state = json.loads((path / "agent/state.json").read_text())
    assert state["pending"]["name"] == "agent_001"
    monkeypatch.setattr(module, "run_trial", actual)
    run_agent(path, "test-only-scripted-planner", 1, True, planner)
    assert len(calls) == 1


def test_stop_action_keeps_best_fixed_trial(run):
    path, _ = run
    result = run_agent(path, "test-only-scripted-planner", planner=lambda obs:
                       Decision(action="stop", assessment="先保留基线", rationale="不继续搜索", settings=None))
    assert result["trials_completed"] == 0
    assert result["selected_trial"] == result["best_fixed_trial"]
    assert not (path / "seed_0/agent_001").exists()


def test_observation_does_not_read_test_metrics(run):
    path, cfg = run
    state = {"trials": []}
    first = observation(path, cfg, state, 3, True)
    (path / "evaluation_with_test").mkdir()
    (path / "evaluation_with_test/summary.csv").write_text("SECRET_TEST_SCORE_SENTINEL")
    second = observation(path, cfg, state, 3, True)
    assert first == second
    text = json.dumps(first)
    assert "SECRET_TEST_SCORE_SENTINEL" not in text
    assert "delta" not in first and "data_dir" not in first


def test_llm_uses_real_sdk_structured_output_without_network():
    settings = dict(learning_rate=.001, weight_decay=.0001, kd_weight=.3,
                    peer_weight=.1, contrast_weight=.02, teacher_weights=[.25]*4)
    expected = decision(settings)
    requests = []
    def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 1, "status": "completed",
            "model": "test-model", "error": None, "incomplete_details": None,
            "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": expected.model_dump_json(), "annotations": []}]}],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
        })
    planner = LLMPlanner.__new__(LLMPlanner)
    planner.client = OpenAI(api_key="unit-test-only", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(serve)))
    planner.model, planner.last_call = "test-model", None
    actual = planner({"trials": [], "remaining_trials": 1})
    assert actual == expected
    assert requests[0]["store"] is False
    assert requests[0]["text"]["format"]["type"] == "json_schema"
    assert requests[0]["text"]["format"]["strict"] is True
    assert planner.last_call["usage"]["total_tokens"] == 20


@pytest.mark.parametrize("weights", [[.5, .5], [-1, 1, 1, 0], [0, 0, 0, 0], [float("nan"), 0, 0, 1]])
def test_llm_teacher_weight_boundary(weights):
    settings = dict(learning_rate=.001, weight_decay=.0001, kd_weight=.3,
                    peer_weight=.1, contrast_weight=.02, teacher_weights=weights)
    with pytest.raises(ValueError):
        validate_decision(decision(settings), 4)

