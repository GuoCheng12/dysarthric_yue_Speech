from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import launch_real_curriculum_v2 as launcher


@pytest.mark.parametrize("dynamic_only", [False, True])
def test_launch_detaches_and_never_records_credentials(tmp_path, monkeypatch, dynamic_only):
    runner = launcher.runner
    monkeypatch.setattr(runner, "verify_frozen", lambda root: {})
    runner.save(tmp_path / "readiness.json", {
        "offline_checks_passed": True, "real_audio_backward_passed": True})
    runner.save(tmp_path / "protocol.json", {})
    if dynamic_only:
        runner.save(tmp_path / "dynamic_restart.json", {"input_sha256": {}})
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    assert launcher.launch(tmp_path, dynamic_only=dynamic_only) == 0
    assert captured["start_new_session"] is True
    assert captured["close_fds"] is True
    assert captured["stdin"] == launcher.subprocess.DEVNULL
    assert captured["stderr"] == launcher.subprocess.STDOUT
    assert captured["env"]["OPENAI_API_KEY"] == "unit-test-placeholder"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"]
    assert captured["command"][-1] == "--worker"
    assert ("--dynamic-only" in captured["command"]) is dynamic_only
    assert "unit-test-placeholder" not in (tmp_path / "background_launch.json").read_text()
    assert "unit-test-placeholder" not in (tmp_path / "background.log").read_text()


def test_worker_records_completion(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(launcher.runner, "run", seen.append)
    assert launcher.worker(tmp_path) == 0
    assert seen == [tmp_path]
    assert launcher.runner.read(tmp_path / "background_exit.json")["status"] == "complete"


def test_worker_records_failure_without_retry(tmp_path, monkeypatch):
    calls = []

    def fail(root):
        calls.append(root)
        raise RuntimeError("intentional test failure")

    monkeypatch.setattr(launcher.runner, "run", fail)
    assert launcher.worker(tmp_path) == 1
    assert calls == [tmp_path]
    receipt = launcher.runner.read(tmp_path / "background_exit.json")
    assert receipt["exit_code"] == 1
    assert receipt["exception_type"] == "RuntimeError"


def test_spawn_arm_pins_one_gpu(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    launcher.spawn_arm(tmp_path, "random", "GPU-random-card")
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-random-card"
    assert captured["command"][-2:] == ["--arm", "random"]
    assert captured["start_new_session"] is True


def test_parallel_scheduler_keeps_static_dynamic_on_same_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.runner, "verify_frozen", lambda root: {})
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    calls, completed, reports = [], [], []

    def spawn(root, arm, gpu):
        if arm == "dynamic":
            assert "static" in completed
        calls.append((arm, gpu))

        def poll():
            completed.append(arm)
            return 0

        return SimpleNamespace(pid=len(calls), poll=poll)

    monkeypatch.setattr(launcher, "spawn_arm", spawn)
    monkeypatch.setattr(launcher.runner, "report", reports.append)
    launcher.run_parallel(tmp_path, ["GPU-A", "GPU-B"])
    assert calls == [("static", "GPU-A"), ("random", "GPU-B"), ("dynamic", "GPU-A")]
    assert reports == [tmp_path]


def test_parallel_scheduler_stops_sibling_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.runner, "verify_frozen", lambda root: {})
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    processes = {"static": SimpleNamespace(pid=10, poll=lambda: 1),
                 "random": SimpleNamespace(pid=11, poll=lambda: None)}
    monkeypatch.setattr(launcher, "spawn_arm", lambda root, arm, gpu: processes[arm])
    stopped = []
    monkeypatch.setattr(launcher, "stop_arm", lambda process: stopped.append(process.pid))
    with pytest.raises(launcher.subprocess.CalledProcessError):
        launcher.run_parallel(tmp_path, ["GPU-A", "GPU-B"])
    assert set(stopped) == {10, 11}


@pytest.mark.parametrize("exit_code", [0, 1])
def test_dynamic_restart_launches_no_controls_and_never_retries(tmp_path, monkeypatch, exit_code):
    monkeypatch.setattr(launcher.runner, "verify_frozen", lambda root: {})
    launcher.runner.save(tmp_path / "dynamic_restart.json", {"input_sha256": {}})
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-A")
    calls, reports = [], []
    process = SimpleNamespace(pid=10, wait=lambda: exit_code, poll=lambda: exit_code)

    def spawn(root, arm, gpu):
        calls.append((arm, gpu))
        return process

    monkeypatch.setattr(launcher, "spawn_arm", spawn)
    monkeypatch.setattr(launcher.runner, "report", reports.append)
    assert launcher.worker(tmp_path, dynamic_only=True) == exit_code
    assert calls == [("dynamic", "GPU-A")]
    assert reports == ([tmp_path] if exit_code == 0 else [])
    assert launcher.runner.read(tmp_path / "exit_dynamic.json")["exit_code"] == exit_code


def test_dynamic_restart_rejects_modified_shared_input_before_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.runner, "verify_frozen", lambda root: {})
    first = tmp_path / "first_course.json"
    launcher.runner.save(first, {"fixture": True})
    launcher.runner.save(tmp_path / "dynamic_restart.json", {"input_sha256": {str(first): "wrong"}})
    monkeypatch.setattr(launcher, "spawn_arm", lambda *args: pytest.fail("must not start training"))
    with pytest.raises(AssertionError):
        launcher.run_dynamic(tmp_path)
