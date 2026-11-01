"""CLI state transitions only: no mocked model result counts as a live smoke."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "mas_dpo_loop_cli_under_test",
    Path(__file__).resolve().parents[1] / "examples/mas_asr_dpo_loop.py",
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def journal(tmp_path, monkeypatch):
    current = SimpleNamespace(phase="initialized", round_index=0)
    protocol = SimpleNamespace(round_count=2, preference_loop=SimpleNamespace(comparison="four_arms", execution_version="serial_v1"))
    events = []

    class MemoryStore:
        def __init__(self, _root):
            pass

        def current(self):
            return current

        def protocol(self):
            return protocol

    monkeypatch.setattr(cli, "RunStore", MemoryStore)
    monkeypatch.setattr(cli, "preflight", lambda _root: None)
    return SimpleNamespace(root=tmp_path, current=current, protocol=protocol, events=events)


@pytest.mark.parametrize("phase,round_index", [("committed", 1), ("assessed", 2), ("proposed", 2), ("trained", 2), ("reflected", 2)])
def test_run_resumes_current_or_next_round_instead_of_round_one(journal, monkeypatch, capsys, phase, round_index):
    journal.current.phase, journal.current.round_index = phase, round_index
    calls = []
    monkeypatch.setattr(cli, "run_round", lambda root, number: calls.append((root, number)) or True)
    cli.main(["run", str(journal.root)])
    assert calls == [(journal.root, 2)]
    assert "complete_fixed_rounds" in capsys.readouterr().out


@pytest.mark.parametrize("wrong_round", [1, 99])
def test_explicit_round_must_match_journal_even_beyond_loop_range(journal, wrong_round):
    journal.current.phase, journal.current.round_index = "assessed", 2
    with pytest.raises(ValueError, match="round"):
        cli.main(["run", str(journal.root), "--round", str(wrong_round)])


def install_round_stages(journal, monkeypatch, *, has_pairs=True):
    def stage(name, next_phase):
        def execute(_root, number):
            journal.events.append(name)
            journal.current.phase, journal.current.round_index = next_phase, number
        return execute

    monkeypatch.setattr(cli.loop, "assess", stage("assess", "assessed"))
    monkeypatch.setattr(cli.loop, "prescribe", stage("prescribe", "materialized"))
    monkeypatch.setattr(cli.loop, "generate", stage("generate", "materialized"))
    monkeypatch.setattr(cli.loop, "reassess", stage("reassess", "reassessed"))
    monkeypatch.setattr(cli.loop, "reflect", stage("reflect", "committed"))

    def sample(_root, number):
        journal.events.append("sample")
        journal.current.phase = "synthesized" if has_pairs else "materialized"
        return has_pairs

    monkeypatch.setattr(cli.loop, "sample_and_pair", sample)


def test_proposed_round_finishes_materialization_without_new_assessment(journal, monkeypatch):
    journal.current.phase, journal.current.round_index = "proposed", 2
    install_round_stages(journal, monkeypatch)
    monkeypatch.setattr(cli, "finish_update", lambda _root, _number: journal.events.append("finish_update"))
    assert cli.run_round(journal.root, 2)
    assert journal.events == ["prescribe", "generate", "sample", "finish_update", "reassess", "reflect"]
    assert journal.current.phase == "committed"


def test_reflected_round_only_commits_existing_reflection(journal, monkeypatch):
    journal.current.phase, journal.current.round_index = "reflected", 2
    install_round_stages(journal, monkeypatch)
    monkeypatch.setattr(cli, "finish_update", lambda _root, _number: pytest.fail("must not repeat training"))
    assert cli.run_round(journal.root, 2)
    assert journal.events == ["reflect"]


def test_all_pairless_stops_dpo_but_finishes_sft_controls_and_all_evaluations(journal, monkeypatch):
    journal.current.phase, journal.current.round_index = "materialized", 1
    install_round_stages(journal, monkeypatch, has_pairs=False)
    updated, skipped, evaluated = [], [], []

    def train(root, number, arm):
        directory = cli.loop.round_dir(root, number) / arm
        directory.mkdir(parents=True)
        if arm.endswith("dpo"):
            skipped.append(arm)
            (directory / "no_update.json").write_text('{"training_executed":false}')
            return False
        updated.append(arm)
        (directory / "training").mkdir()
        (directory / "training/training_result.json").write_text('{"status":"complete"}')
        return True

    def evaluate(root, number, arm):
        evaluated.append(arm)
        directory = cli.loop.round_dir(root, number) / arm / "evaluation"
        directory.mkdir()
        (directory / "predictions.jsonl").write_text("{}\n")

    monkeypatch.setattr(cli.loop, "train_branch", train)
    monkeypatch.setattr(cli.loop, "evaluate_branch", evaluate)
    assert cli.run_round(journal.root, 1) is False
    assert updated == ["r_sft", "mas_sft"]
    assert skipped == ["r_dpo", "mas_dpo"]
    assert evaluated == list(cli.loop.ARMS)
    assert "reflect" not in journal.events and "reassess" not in journal.events
    assert journal.current.phase == "materialized"


def test_unknown_phase_is_an_error_not_no_preference_pairs(journal):
    journal.current.phase, journal.current.round_index = "unknown_phase", 2
    with pytest.raises(RuntimeError, match="unknown_phase"):
        cli.run_round(journal.root, 2)


def test_completed_journal_is_a_noop(journal, monkeypatch, capsys):
    journal.current.phase, journal.current.round_index = "committed", 2
    monkeypatch.setattr(cli, "preflight", lambda *_args: pytest.fail("completed run does not need credentials"))
    monkeypatch.setattr(cli, "run_round", lambda *_args: pytest.fail("completed run must not execute another round"))
    cli.main(["run", str(journal.root)])
    assert "complete_fixed_rounds" in capsys.readouterr().out


@pytest.fixture
def credential_protocol(tmp_path, monkeypatch):
    protocol = SimpleNamespace(
        agent=SimpleNamespace(api_key_env="TEST_MAS_API_KEY"),
        preference_loop=SimpleNamespace(audio_review_api_key_env="TEST_MAS_API_KEY"),
    )
    monkeypatch.setattr(cli, "RunStore", lambda _root: SimpleNamespace(protocol=lambda: protocol))
    monkeypatch.setattr(cli.os, "environ", dict(cli.os.environ))
    for key in ("TEST_MAS_API_KEY", "TEST_AUDIO_API_KEY"):
        cli.os.environ.pop(key, None)
    return SimpleNamespace(root=tmp_path, protocol=protocol)


def test_existing_credential_never_prompts(credential_protocol, monkeypatch):
    monkeypatch.setenv("TEST_MAS_API_KEY", "test-existing-value")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.getpass, "getpass", lambda *_args: pytest.fail("existing credential must be reused"))
    cli.preflight(credential_protocol.root)
    assert cli.os.environ["TEST_MAS_API_KEY"] == "test-existing-value"


@pytest.mark.parametrize("separate_audio_key", [False, True])
def test_interactive_credentials_are_in_memory_and_shared_key_is_prompted_once(
    credential_protocol, monkeypatch, capsys, separate_audio_key
):
    if separate_audio_key:
        credential_protocol.protocol.preference_loop.audio_review_api_key_env = "TEST_AUDIO_API_KEY"
    prompts = []

    def enter_key(prompt):
        prompts.append(prompt)
        return "test-audio-value" if prompt.startswith("TEST_AUDIO_API_KEY") else "test-mas-value"

    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.getpass, "getpass", enter_key)
    cli.preflight(credential_protocol.root)
    assert len(prompts) == (2 if separate_audio_key else 1)
    assert cli.os.environ["TEST_MAS_API_KEY"] == "test-mas-value"
    if separate_audio_key:
        assert cli.os.environ["TEST_AUDIO_API_KEY"] == "test-audio-value"
    assert list(credential_protocol.root.iterdir()) == []
    assert capsys.readouterr() == ("", "")


def test_noninteractive_preflight_names_only_missing_credentials(credential_protocol, monkeypatch):
    credential_protocol.protocol.preference_loop.audio_review_api_key_env = "TEST_AUDIO_API_KEY"
    monkeypatch.setenv("TEST_MAS_API_KEY", "test-existing-value")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.getpass, "getpass", lambda *_args: pytest.fail("noninteractive run must not prompt"))
    with pytest.raises(RuntimeError) as caught:
        cli.preflight(credential_protocol.root)
    assert str(caught.value).endswith("TEST_AUDIO_API_KEY")
    assert "TEST_MAS_API_KEY" not in str(caught.value)
    assert "test-existing-value" not in str(caught.value)


def test_empty_interactive_credential_is_still_missing(credential_protocol, monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.getpass, "getpass", lambda *_args: "")
    with pytest.raises(RuntimeError, match="TEST_MAS_API_KEY"):
        cli.preflight(credential_protocol.root)
    assert "TEST_MAS_API_KEY" not in cli.os.environ


def test_prepare_does_not_request_credentials(journal, monkeypatch, capsys):
    monkeypatch.setattr(cli, "preflight", lambda *_args: pytest.fail("prepare does not need credentials"))
    monkeypatch.setattr(cli.loop, "prepare", lambda *_args, **_kwargs: None)
    cli.main(["prepare", str(journal.root), "--audio-review-model", "test-model",
              "--audio-review-base-url", "https://example.invalid"])
    assert "prepared_only" in capsys.readouterr().out


def test_dpo_only_run_updates_and_evaluates_only_frozen_branches(journal, monkeypatch):
    journal.protocol.preference_loop.comparison = "dpo_only"
    install_round_stages(journal, monkeypatch)
    calls = []
    monkeypatch.setattr(cli.loop, "train_branch", lambda root, number, arm: calls.append(("train", arm)))
    monkeypatch.setattr(cli.loop, "evaluate_branch", lambda root, number, arm: calls.append(("evaluate", arm)))
    assert cli.run_round(journal.root, 1)
    assert calls == [("train", "r_dpo"), ("evaluate", "r_dpo"),
                     ("train", "mas_dpo"), ("evaluate", "mas_dpo")]
    assert journal.current.phase == "committed"


def test_dpo_only_selection_is_frozen_during_prepare(journal, monkeypatch):
    options = {}
    monkeypatch.setattr(cli.loop, "prepare", lambda root, **kwargs: options.update(kwargs))
    cli.main(["prepare", str(journal.root), "--dpo-only"])
    assert options["comparison"] == "dpo_only"
    with pytest.raises(SystemExit):
        cli.parse_args(["run", str(journal.root), "--dpo-only"])


def test_comparison_default_preserves_historical_protocol_serialization():
    from therapist_harness.preference_contract import PreferenceLoopConfig
    config = PreferenceLoopConfig(audio_review_model="fixture", audio_review_base_url="https://example.invalid")
    assert "comparison" not in config.model_dump(mode="json")
    selected = config.model_copy(update={"comparison": "dpo_only"})
    assert selected.model_dump(mode="json")["comparison"] == "dpo_only"
    assert cli.loop.enabled_arms(SimpleNamespace(preference_loop=selected)) == ("r_dpo", "mas_dpo")


def test_synthetic_only_freezes_one_branch_and_persistent_review(journal, monkeypatch):
    options = {}
    monkeypatch.setattr(cli.loop, "prepare", lambda root, **kwargs: options.update(kwargs))
    cli.main(["prepare", str(journal.root), "--synthetic-only"])
    assert options["comparison"] == "synthetic_only"
    assert options["execution_version"] == "persistent_synthetic_only_v1"
    journal.protocol.preference_loop.comparison = "synthetic_only"
    install_round_stages(journal, monkeypatch)
    calls = []
    monkeypatch.setattr(cli.loop, "train_branch", lambda root, number, arm: calls.append(("train", arm)))
    monkeypatch.setattr(cli.loop, "evaluate_branch", lambda root, number, arm: calls.append(("evaluate", arm)))
    assert cli.run_round(journal.root, 1)
    assert calls == [("train", "mas_dpo"), ("evaluate", "mas_dpo")]


def test_synthetic_only_contract_requires_zero_real_and_unit_loss():
    from pydantic import ValidationError
    from therapist_harness.preference_contract import PreferenceLoopConfig
    options = dict(audio_review_model="fixture", audio_review_base_url="https://example.invalid", comparison="synthetic_only")
    with pytest.raises(ValidationError, match="zero real exposure"):
        PreferenceLoopConfig(**options)
    config = PreferenceLoopConfig(**options, real_per_step=0, synthetic_weight=1.0)
    assert cli.loop.enabled_arms(SimpleNamespace(preference_loop=config)) == ("mas_dpo",)


def test_synthetic_only_windows_have_no_real_exposure_and_unit_total_weight():
    from therapist_harness.dpo_training import SourceBalancedWindows, group_training_rows
    rows = [dict(audio="/fixture/a.wav", audio_id="a", source="synthetic", chosen="目標", rejected="錯一", pair_weight=0.5),
            dict(audio="/fixture/a.wav", audio_id="a", source="synthetic", chosen="目標", rejected="錯二", pair_weight=0.5),
            dict(audio="/fixture/b.wav", audio_id="b", source="synthetic", chosen="另一句", rejected="錯誤", pair_weight=1.0)]
    windows = SourceBalancedWindows(group_training_rows(rows, "dpo"), 0, 2, 1.0, 42)
    batch = windows.next()
    assert windows.counts == {"real": 0, "synthetic": 2}
    assert len(batch) == 3
    assert sum(x["_loss_weight"] for x in batch) == pytest.approx(1.0)
    assert {x["source"] for x in batch} == {"synthetic"}
