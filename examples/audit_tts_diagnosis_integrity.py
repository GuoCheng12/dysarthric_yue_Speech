"""Recheck the recorded provenance of completed diagnosis training runs."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/tts_utility_diagnosis_v1')


def run():
    expected = {}
    conflicts = []
    runs = []

    def add(path, digest):
        if path in expected and expected[path] != digest:
            conflicts.append(path)
        expected[path] = digest

    for family in ['additive', 'post_small']:
        for result in sorted((ROOT / family).glob('*/seed_*/result.json')):
            directory = result.parent
            protocol = json.loads((directory / 'protocol.json').read_text())
            monitor = json.loads((directory / 'monitor.json').read_text())
            assert monitor['status'] == 'complete', directory
            runs.append(str(directory))
            for key in ['source_sha256', 'input_sha256', 'audio_sha256', 'hashes']:
                for path, digest in protocol.get(key, {}).items():
                    add(path, digest)
            if 'train_sha256' in protocol:
                add(str(directory / 'train.jsonl'), protocol['train_sha256'])
            intervention = directory / 'intervention.json'
            if intervention.exists():
                data = json.loads(intervention.read_text())
                if isinstance(data.get('source_sha256'), str):
                    add(data['command'][0], data['source_sha256'])
                if 'real_source_sha256' in data:
                    add(data['real_source'], data['real_source_sha256'])
    mismatches = []
    for name, digest in sorted(expected.items()):
        path = Path(name)
        if not path.is_file():
            mismatches.append({'path': name, 'reason': 'missing'})
            continue
        with path.open('rb') as handle:
            actual = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual != digest:
            mismatches.append({'path': name, 'expected': digest, 'actual': actual})
    output = {
        'time_unix': time.time(), 'completed_runs': len(runs), 'runs': runs,
        'unique_verified_files': len(expected), 'hash_conflicts': conflicts,
        'hash_mismatches': mismatches,
        'scope': 'completed run protocols: source, inputs, audio, ordered training manifests, warm-start weights and intervention hashes; unrecorded files are outside this audit',
    }
    (ROOT / 'integrity_snapshot.json').write_text(json.dumps(output, indent=2) + '\n')
    print({k: v for k, v in output.items() if k != 'runs'}, flush=True)
    assert not conflicts and not mismatches


if __name__ == '__main__':
    run()
