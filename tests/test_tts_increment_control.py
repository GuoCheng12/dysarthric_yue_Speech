import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "tts_increment_control", Path(__file__).resolve().parents[1] / "examples/tts_increment_control.py"
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_matched_selection():
    real = control.rows(control.REAL)
    refs = control.read(control.REFS)["references"]
    retained, held, replay = control.select_split(real, refs)
    assert (len(retained), len(held), len(replay)) == (1541, 576, 576)
    assert not {r["utt_id"] for r in retained} & {r["utt_id"] for r in held}
    assert {r["reference_wav"] for r in refs} <= {r["audio"] for r in retained}
    assert {r["speaker_id"] for r in retained} == {r["speaker_id"] for r in real}
    for target, repeated in zip(held, replay):
        assert (target["speaker_id"], target["zero_shot_bucket"]) == (
            repeated["speaker_id"], repeated["zero_shot_bucket"])
        assert repeated in retained
    assert control.select_split(real, refs) == (retained, held, replay)


def test_training_commands_identical_except_paths_and_seed(tmp_path):
    configs = []
    for arm in control.ARMS:
        for seed in control.SEEDS:
            command = control.training_command(tmp_path, arm, seed)
            cfg = dict(zip(command[2::2], command[3::2]))
            assert "--lora_init_adapter_path" not in cfg and "--resume" not in cfg
            assert cfg["--batch_size"] == "8" and cfg["--grad_acc"] == "4"
            assert cfg["--lr"] == "2e-4" and cfg["--max_steps"] == "140"
            assert cfg["--save_steps"] == "50"
            for key in ("--train_file", "--output_dir", "--seed"):
                cfg.pop(key)
            configs.append(cfg)
    assert all(c == configs[0] for c in configs)
