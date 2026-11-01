"""Deterministic data exposure for the fixed-budget real-data comparison."""

from collections import Counter
import math
import random


class ExposureSampler:
    """One RNG independent of model/dropout RNG; no batches are prefetched."""

    def __init__(self, train_ids, groups, seed=42):
        self.train_ids = list(train_ids)
        assert len(set(self.train_ids)) == len(self.train_ids) > 0
        self.groups = {key: list(ids) for key, ids in groups.items()}
        allowed = set(self.train_ids)
        assert self.groups and all(ids and len(ids) == len(set(ids))
                                   and set(ids) <= allowed for ids in self.groups.values())
        self.rng = random.Random(seed)
        self.pending = []
        self.sweeps = 0
        self.exposures = 0

    def draw(self, count, group_weights=None):
        assert count > 0
        if group_weights is None:
            result = []
            while len(result) < count:
                if not self.pending:
                    self.pending = self.train_ids.copy()
                    self.rng.shuffle(self.pending)
                    self.sweeps += 1
                take = min(count - len(result), len(self.pending))
                result.extend(self.pending[:take])
                del self.pending[:take]
        else:
            assert set(group_weights) == set(self.groups)
            values = [float(group_weights[key]) for key in self.groups]
            assert all(math.isfinite(value) and value >= 0 for value in values)
            assert math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-6)
            keys = list(self.groups)
            result = []
            for _ in range(count):
                key = self.rng.choices(keys, weights=values, k=1)[0]
                result.append(self.rng.choice(self.groups[key]))
        self.exposures += count
        return result

    def state_dict(self):
        return dict(rng=self.rng.getstate(), pending=self.pending.copy(),
                    sweeps=self.sweeps, exposures=self.exposures)

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])
        self.pending = list(state['pending'])
        self.sweeps = state['sweeps']
        self.exposures = state['exposures']


def exposure_summary(utterance_ids, train_by_id):
    counts = Counter(utterance_ids)
    return dict(actual_audio_exposures=len(utterance_ids), unique_audio=len(counts),
                per_utterance=dict(sorted(counts.items())),
                per_patient=dict(sorted(Counter(train_by_id[uid]['speaker_id']
                                                for uid in utterance_ids).items())),
                audio_seconds=sum(float(train_by_id[uid]['duration']) for uid in utterance_ids))
