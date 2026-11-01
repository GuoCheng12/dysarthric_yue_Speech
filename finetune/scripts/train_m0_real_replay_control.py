"""Matched 67-update real-replay-only control for the first real MAS pilot."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import time

import torch

import qwen3_asr_lora_sft as training
from train_therapist_round import RoundMonitor
from train_setting_d_m0 import decode

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from therapist_harness.evidence import evaluate_snapshot  # noqa: E402
from therapist_harness.store import sha256_file, sha256_directory, write_immutable_json, write_immutable_bytes  # noqa: E402

PILOT = Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/therapist_m0_b8g4_pilot_v1')
REAL = Path('/data/qwen3-asr/finetune/data_setting_d_v1/setting_d_train.jsonl')
TEST = Path('/data/qwen3-asr/finetune/data_setting_d_merged_test_v1/setting_d_merged_test.jsonl')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def control_command(source, train_path, output_path):
    command = list(source)
    command[command.index('--train_file') + 1] = str(train_path)
    command[command.index('--output_dir') + 1] = str(output_path)
    return command


class ControlMonitor(RoundMonitor):
    def __init__(self, root, scope):
        super().__init__(root, 67)
        self.scope = scope
        self.augmented = {
            int(row['model'].removeprefix('round_step')): row['metrics']
            for row in json.loads((PILOT/'merged_test_reassessment_v1/results.json').read_text())
            if row['model'].startswith('round_step')
        }

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        self.state.update(status='evaluating_merged_test', step=step)
        self.publish()
        torch.cuda.empty_cache()
        start = time.monotonic()
        checkpoint = Path(args.output_dir) / f'checkpoint-{step}'
        predicted = {}
        # The two files are decoding shards of ONE 511-row test. This preserves
        # the inference batch boundaries of the already evaluated treatment arm.
        for shard in ('dev', 'test'):
            destination = self.root / f'evaluation/step_{step}_{shard}_shard'
            decode(checkpoint, shard, destination, self.root / f'evaluation/step_{step}_{shard}_shard.log')
            with (destination/'predictions.csv').open(newline='') as handle:
                for row in csv.DictReader(handle):
                    assert row['utt_id'] not in predicted
                    predicted[row['utt_id']] = row
        assert set(predicted) == {row['utt_id'] for row in self.scope}
        rows = []
        for reference in self.scope:
            prediction = predicted[reference['utt_id']]
            for field in ('speaker_id', 'prompt_id', 'clean_gt', 'zero_shot_bucket'):
                assert prediction[field] == reference[field]
            rows.append({**reference, 'prediction': prediction['prediction']})
        metrics = evaluate_snapshot(rows, prediction_column='prediction', critical_cer_threshold=.5).model_dump(mode='json')
        destination = self.root / f'evaluation/merged_test_step_{step}'
        write_immutable_bytes(destination/'predictions.jsonl', ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
        write_immutable_json(destination/'summary.json', {
            'checkpoint':str(checkpoint), 'test_path':str(TEST), 'test_sha256':sha256_file(TEST),
            'prediction_count':len(rows), 'metrics':metrics})
        item = {'step':step, 'metrics':metrics, 'augmented_metrics':self.augmented[step],
                'augmented_minus_control_overall_cer':self.augmented[step]['overall']['mean_utterance_cer']-metrics['overall']['mean_utterance_cer'],
                'decode_seconds':time.monotonic()-start}
        self.history.append(item)
        self.state.update(status='training')
        self.publish()
        print('[replay-control] '+json.dumps(item),flush=True)
        return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    execution = json.loads((PILOT/'training_execution.json').read_text())
    protocol = json.loads((PILOT/'protocol.json').read_text())
    parent = Path(protocol['initial_adapter']['path'])
    assert sha256_directory(parent) == protocol['initial_adapter']['sha256']
    for source in (Path(training.__file__), REPO/'finetune/official/qwen3_asr_sft.py'):
        assert sha256_file(source) == execution['source_sha256'][str(source)]
    assert sha256_file(REAL) == protocol['real_train']['sha256']
    assert sha256_file(TEST) == '189d749e7e65f14a0440d7f6707a0e1dcfa1a5a62c6c0e5f362b08c1c53e2f98'
    real = [{k:row[k] for k in ('audio','text','prompt')} for row in read_jsonl(REAL)]
    augmented = read_jsonl(PILOT/'train.jsonl')
    assert len(real) == 2117 and len(augmented) == 2141 and real == augmented[:2117]
    write_immutable_bytes(root/'train.jsonl',''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in real).encode())
    command = control_command(execution['command'],root/'train.jsonl',root/'training')
    original_args = vars(training.parse_args(execution['command'][2:]))
    control_args = vars(training.parse_args(command[2:]))
    assert {k for k in original_args if original_args[k] != control_args[k]} == {'train_file','output_dir'}
    assert control_args['max_steps'] == 67 and control_args['lora_init_adapter_path'] == str(parent)
    assert not control_args['resume'] and not control_args['resume_from']
    scope = read_jsonl(TEST)
    assert len(scope) == len({row['utt_id'] for row in scope}) == 511
    for field in ('utt_id','prompt_id'):
        assert not {r[field] for r in read_jsonl(REAL)} & {r[field] for r in scope}
    config = {'experiment':'matched_real_replay_continuation_control', 'parent_checkpoint':str(parent),
        'parent_sha256':protocol['initial_adapter']['sha256'], 'real_rows':2117,'synthetic_rows':0,
        'test_rows':511,'test_sha256':sha256_file(TEST),'train_sha256':sha256_file(root/'train.jsonl'),
        'command':command,'training_argument_differences':['train_file','output_dir'],
        'optimizer_reset':True,'scheduler_reset':True,'fixed_endpoint_steps':67,'checkpoint_selection':False,
        'evaluation_shards':'original dev + original test, same batch boundaries as treatment; aggregate all 511',
        'source_sha256':{str(p):sha256_file(p) for p in (Path(__file__),Path(training.__file__),REPO/'finetune/official/qwen3_asr_sft.py',REPO/'finetune/scripts/train_therapist_round.py',REPO/'finetune/scripts/train_setting_d_m0.py')}}
    write_immutable_json(root/'run_config.json',config)
    monitor = ControlMonitor(root,scope)
    monitor.publish()

    class SupervisedTrainer(training.QwenASRLoRATrainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.add_callback(monitor)

    training.QwenASRLoRATrainer = SupervisedTrainer
    sys.argv = command[1:]
    try:
        training.main()
        assert [h['step'] for h in monitor.history] == [10,20,30,40,50,60,67]
        state = json.loads((root/'training/checkpoint-67/trainer_state.json').read_text())
        assert state['global_step'] == 67 and state['epoch'] == 1.0
        for path,digest in config['source_sha256'].items():
            assert sha256_file(Path(path)) == digest
        assert sha256_directory(parent) == config['parent_sha256']
        assert sha256_file(root/'train.jsonl') == config['train_sha256']
        summary = {'status':'complete','scope':'2117 train / 511 test', 'real_rows':2117,'synthetic_rows':0,
            'new_optimizer_steps':67,'checkpoint':str(root/'training/checkpoint-67'),
            'checkpoint_sha256':sha256_directory(root/'training/checkpoint-67'),
            'history':monitor.history}
        write_immutable_json(root/'run_summary.json',summary)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(['new_step','real_only_CER_pct','plus24_CER_pct','plus24_minus_real_pp'])
        for row in monitor.history:
            writer.writerow([row['step'],row['metrics']['overall']['mean_utterance_cer']*100,
                row['augmented_metrics']['overall']['mean_utterance_cer']*100,
                row['augmented_minus_control_overall_cer']*100])
        write_immutable_bytes(root/'comparison.csv',buffer.getvalue().encode())
        monitor.state.update(status='complete')
        monitor.publish()
    except Exception as error:
        monitor.state.update(status='failed',error=f'{type(error).__name__}: {error}')
        monitor.publish()
        raise


if __name__ == '__main__':
    main()
