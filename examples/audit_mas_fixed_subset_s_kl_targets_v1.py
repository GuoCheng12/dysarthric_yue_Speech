"""Use existing paired EvidencePack analysis to test the intended real targets."""
import gzip
import json
from pathlib import Path
import sys

REPO=Path('/data/qwen3-asr/repo/dysarthric_yue_Speech')
sys.path.insert(0,str(REPO/'src'))
from therapist_harness.evidence import compare_prediction_snapshots,_key_descriptor
from therapist_harness.store import sha256_file,write_immutable_json
ROOT=Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1/fixed_subset_S_DPO_real_KL_v1')
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(x) for x in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if x.strip()]

def main():
    paths={'initial':Path('/tmp/mas_deterministic_R_only_eval_v1/evaluation/predictions.jsonl.gz'),
        'full_R_SG':Path('/data/qwen3-asr/records/therapist_mas_setting_d_v1/mas_design_exploration_20260908_v1/synthetic_utility_allocation_DPO_v1/uniform/evaluation/predictions.jsonl.gz'),
        'S_KL':ROOT/'S_KL/evaluation/predictions.jsonl.gz'}
    data={a:rows(p) for a,p in paths.items()}
    descriptors=[_key_descriptor(('substitution',level,before,after,None,None)) for level,before,after in [('nucleus','i','yu'),('nucleus','i','e'),('coda','m','n')]]
    proposal_path=Path('/tmp/mas_kl_dpo_loop_v3_formal96/round_001/proposal.json')
    proposal=read(proposal_path);ids={x['subject_evidence_id'] for x in proposal['expected_outcomes']}
    assert all(d.evidence_id in ids for d in descriptors)
    proto=Path('/tmp/mas_kl_dpo_loop_v3_formal96/protocol.json')
    comparisons={}
    for before,after in [('initial','S_KL'),('full_R_SG','S_KL')]:
        result=compare_prediction_snapshots(data[before],data[after],before_prediction_column='prediction',after_prediction_column='prediction',
            critical_cer_threshold=read(proto)['evaluation']['critical_cer_threshold'],monitored_clusters=descriptors,off_target_max_per_level_per_direction=2)
        comparisons[before+'_to_'+after]=dict(monitored_clusters=[x.model_dump(mode='json') for x in result.monitored_cluster_changes],
            off_target_changes=[x.model_dump(mode='json') for x in result.off_target_changes],
            phonology_coverage_before=result.phonology_coverage_before.model_dump(mode='json'),
            phonology_coverage_after=result.phonology_coverage_after.model_dump(mode='json'))
    files=[Path(__file__),REPO/'src/therapist_harness/evidence.py',proposal_path,proto,*paths.values()]
    out=dict(status='complete_real_paired_target_recovery',comparisons=comparisons,
        scope='Existing conservative real-transcript phonology and paired same-carrier recovery, repeated511 development set. Not synthetic acoustic fidelity or clinical phoneme measurement. Evaluation critical-count definition unchanged; no SG review or admission threshold added.',
        source_sha256={str(f):sha256_file(f) for f in files})
    write_immutable_json(ROOT/'real_target_recovery.json',out)
    print(json.dumps({k:[dict(id=c['descriptor']['evidence_id'],before=c['before']['error_count'],after=c['after']['error_count'],target_positions={f:c['target_positions'][f] for f in ['before_error_count','after_error_count','recovered_count','regressed_count','before_unresolved_count','after_unresolved_count']}) for c in v['monitored_clusters']] for k,v in comparisons.items()},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
