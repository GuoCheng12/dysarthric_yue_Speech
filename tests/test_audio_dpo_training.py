import math
from types import SimpleNamespace

import pytest
import torch

from therapist_harness.dpo_training import (
    AudioPreferenceCollator, SourceBalancedWindows, dpo_losses,
    frozen_reference_log_probs, group_training_rows, transcript_log_probs,
)


def test_sequence_log_probs_mask_prompt_audio_padding_and_sum_tokens():
    torch.manual_seed(42)
    logits = torch.randn(2, 7, 11, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, 2, 3, 4, -100], [-100, -100, 5, 6, -100, -100, -100]])
    log_probs, counts = transcript_log_probs(logits, labels)
    expected = torch.stack([
        sum(logits.log_softmax(-1)[i, pos - 1, token] for pos, token in enumerate(labels[i]) if token != -100)
        for i in range(2)
    ])
    torch.testing.assert_close(log_probs, expected)
    assert counts.tolist() == [3, 2]
    log_probs.sum().backward()
    assert logits.grad[0, :2].count_nonzero() == 0
    assert logits.grad[0, 5:].count_nonzero() == 0
    with pytest.raises(ValueError, match="supervised token"):
        transcript_log_probs(logits, torch.full_like(labels, -100))


def test_standard_dpo_reference_detached_log2_and_correct_gradient_direction():
    chosen = torch.tensor([-2.0, -7.0], requires_grad=True)
    rejected = torch.tensor([-5.0, -4.0], requires_grad=True)
    ref_chosen = chosen.detach().clone().requires_grad_()
    ref_rejected = rejected.detach().clone().requires_grad_()
    losses = dpo_losses(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
    torch.testing.assert_close(losses, torch.full((2,), math.log(2)))
    losses.mean().backward()
    assert torch.all(chosen.grad < 0)
    assert torch.all(rejected.grad > 0)
    assert ref_chosen.grad is None and ref_rejected.grad is None
    improved = dpo_losses(chosen.detach() + 0.2, rejected.detach() - 0.2,
                          ref_chosen, ref_rejected, beta=0.1)
    assert torch.all(improved < losses)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 7))

    def forward(self, x):
        return {"logits": x @ self.weight}


def test_reference_requires_frozen_eval_and_does_not_mutate_or_get_gradients():
    model = ToyModel()
    batch = {"x": torch.randn(2, 4, 3), "labels": torch.tensor([[-100, -100, 2, 3], [-100, 1, 4, -100]])}
    with pytest.raises(ValueError, match="frozen eval"):
        frozen_reference_log_probs(model, batch)
    model.eval().requires_grad_(False)
    before = model.weight.detach().clone()
    result = frozen_reference_log_probs(model, batch)
    assert not result.requires_grad and model.weight.grad is None and not model.training
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


def training_rows():
    rows = []
    for source, count in [("real", 3), ("synthetic", 2)]:
        for audio in range(count):
            wrong_count = audio + 1
            for index in range(wrong_count):
                rows.append({"audio_id": f"{source}{audio}", "audio": f"/{source}{audio}.wav", "source": source,
                             "chosen": "甲", "rejected": str(index), "pair_weight": 1 / wrong_count})
    return rows


def test_audio_balancing_and_real_order_shared_between_real_and_mixed():
    rows = training_rows()
    mixed = SourceBalancedWindows(group_training_rows(rows, "dpo"), 5, 3, 0.05, 42)
    real = SourceBalancedWindows(group_training_rows([r for r in rows if r["source"] == "real"], "dpo"), 5, 3, 0.05, 42)
    for _ in range(4):
        mixed_window, real_window = mixed.next(), real.next()
        assert [r for r in mixed_window if r["source"] == "real"] == real_window
        assert sum(r["_loss_weight"] for r in mixed_window if r["source"] == "real") == pytest.approx(1)
        assert sum(r["_loss_weight"] for r in mixed_window if r["source"] == "synthetic") == pytest.approx(0.05)
        # Every audio occurrence has equal total weight despite differing error counts.
        for occurrence in range(5):
            assert sum(r["_loss_weight"] for r in mixed_window if r["source"] == "real" and r["_occurrence"] == occurrence) == pytest.approx(1 / 5)


def test_explicit_synthetic_schedule_preserves_real_stream_and_audio_mean():
    groups=group_training_rows(training_rows(),'dpo')
    schedule=[['synthetic1','synthetic1','synthetic0'],['synthetic0']*3]
    selected=SourceBalancedWindows(groups,5,3,.05,42,synthetic_schedule=schedule)
    baseline=SourceBalancedWindows(groups,5,3,.05,42)
    for requested in schedule:
        window=selected.next(); other=baseline.next()
        assert [x for x in window if x['source']=='real']==[x for x in other if x['source']=='real']
        actual={x['_occurrence']:x['audio_id'] for x in window if x['source']=='synthetic'}
        assert list(actual.values())==requested
        assert sum(x['_loss_weight'] for x in window if x['source']=='synthetic')==pytest.approx(.05)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 100])
def test_variable_pairs_and_partial_microbatches_have_exact_window_gradient(batch_size):
    window = SourceBalancedWindows(group_training_rows(training_rows(), "dpo"), 3, 2, 0.2, 42).next()
    torch.manual_seed(11)
    x = torch.randn(len(window), 3)
    weights = torch.tensor([r["_loss_weight"] for r in window])
    initial = torch.randn(3)
    expected = initial.clone().requires_grad_()
    actual = initial.clone().requires_grad_()
    zeros = torch.zeros(len(window))
    (dpo_losses(x @ expected, zeros, zeros, zeros, 0.3) * weights).sum().backward()
    for start in range(0, len(window), batch_size):
        end = start + batch_size
        (dpo_losses(x[start:end] @ actual, zeros[start:end], zeros[start:end], zeros[start:end], 0.3)
         * weights[start:end]).sum().backward()
    torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)


def test_sft_collapses_preferences_and_keeps_pairless_audio():
    rows = training_rows() + [{"audio": "/pairless.wav", "chosen": "乙", "source": "real"}]
    groups = group_training_rows(rows, "sft")
    assert len(groups["real"]) == 4
    assert all(len(g) == 1 and g[0]["pair_weight"] == 1 for source in groups.values() for g in source.values())
    with pytest.raises(ValueError, match="distinct"):
        group_training_rows(rows, "dpo")
    with pytest.raises(ValueError, match="duplicate rejected"):
        group_training_rows(training_rows() + [training_rows()[0]], "dpo")


def test_pairless_real_pool_does_not_block_genuine_synthetic_preferences():
    rows = [r for r in training_rows() if r["source"] == "synthetic"]
    windows = SourceBalancedWindows(group_training_rows(rows, "dpo"), 32, 3, 0.05, 42)
    window = windows.next()
    assert windows.counts == {"real": 0, "synthetic": 3}
    assert all(r["source"] == "synthetic" for r in window)
    assert sum(r["_loss_weight"] for r in window) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="no-update"):
        group_training_rows([], "dpo")


class FakeProcessor:
    # EOS == PAD intentionally, so masking by token ID would silently remove EOS.
    tokenizer = SimpleNamespace(eos_token="§", pad_token_id=0)

    def apply_chat_template(self, messages, **kwargs):
        return ["@AUDIO@"]

    def __call__(self, text, audio, **kwargs):
        tokens = [[0 if c == "§" else ord(c) for c in s] for s in text]
        maximum = max(map(len, tokens))
        return {"input_ids": torch.tensor([t + [0] * (maximum - len(t)) for t in tokens]),
                "attention_mask": torch.tensor([[1] * len(t) + [0] * (maximum - len(t)) for t in tokens]),
                "input_features": torch.tensor(audio)}


def test_collator_same_audio_language_conditioning_and_eos_with_shared_pad_id():
    calls = []
    def loader(path):
        calls.append(path)
        return [1.0, 2.0]
    collator = AudioPreferenceCollator(FakeProcessor(), audio_loader=loader)
    result = collator([{"audio": "/one.wav", "chosen": "甲乙", "rejected": "丙", "prompt": ""}])
    assert calls == ["/one.wav"]
    torch.testing.assert_close(result["input_features"][0], result["input_features"][1])
    labels = result["labels"]
    assert labels[0][labels[0].ne(-100)].tolist() == [ord("甲"), ord("乙"), 0]
    assert labels[1][labels[1].ne(-100)].tolist() == [ord("丙"), 0]
    assert labels[1, -1] == -100
