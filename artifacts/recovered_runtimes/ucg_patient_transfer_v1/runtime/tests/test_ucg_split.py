import copy
import pytest
from therapist_harness.sampling_curriculum_data import validate_train_feedback_split,make_sampling_groups,build_sampling_evidence,build_sampling_outcome
from therapist_harness.material_selection_offline import build_index,prepare_candidates,digest

def parts():
 return [dict(utt_id='train',audio='/train.wav',speaker_id='p',clean_gt='木',duration=2,split='train',zero_shot_bucket='medium')], [dict(utt_id='dev',audio='/dev.wav',speaker_id='p',clean_gt='北',duration=2,split='dev',zero_shot_bucket='medium')]

def test_ucg_shared_patient_disjoint_text():
 train,dev=parts();validate_train_feedback_split(train,dev,'D')
 with pytest.raises(ValueError,match='patient-disjoint'):validate_train_feedback_split(train,dev,'E')

@pytest.mark.parametrize('field,value',[('utt_id','train'),('audio','/train.wav'),('clean_gt','木！'),('speaker_id','new-p')])
def test_ucg_rejects_wrong_split(field,value):
 train,dev=parts();dev[0][field]=value
 with pytest.raises(ValueError):validate_train_feedback_split(train,dev,'D')

def test_ucg_index_and_feedback_use_distinct_content():
 train,dev=parts();groups=make_sampling_groups(train)
 predictions=[dict(utt_id='dev',prediction='木')]
 evidence=build_sampling_evidence(train,predictions,groups,feedback_rows=dev,round_index=1,checkpoint_step=0,setting='D')
 assert evidence['setting']=='D' and '完整句子不重叠' in evidence['data_explanation']
 assert evidence['feedback']['current']['audio']==1
 index=build_index(train,dev,{},setting='D')
 assert {p['target_text'] for p in index['profiles'].values()}=={'木'}
 brief=dict(producer='offline_fixture',round_index=1,source_evidence_sha256='0'*64,summary='验证查询',focuses=[dict(focus_id='held',purpose='target',level='text',target_unit='北',reason='无同句录音',evidence_ids=['step0:overall']),dict(focus_id='tone',purpose='target',level='tone',target_unit='6',reason='已有训练声调覆盖',evidence_ids=['step0:overall'])])
 candidates=prepare_candidates(index,brief)
 assert candidates['focuses'][0]['eligible_count']==0
 assert candidates['focuses'][1]['eligible_ids']==['train']
