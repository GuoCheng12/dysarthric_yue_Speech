import copy,json
import pytest
from therapist_harness.ucg_material_context import annotate_ucg_candidates
from therapist_harness.material_selection_offline import build_index,prepare_candidates,build_selector_request,validate_selection,selector_context,digest

def fixture():
 train=[dict(utt_id=p,audio=f'/{p}.wav',speaker_id=p,clean_gt='木頭',duration=2,split='train',zero_shot_bucket='medium') for p in ['p1','p2']]
 dev=[dict(utt_id='dev1',audio='/dev.wav',speaker_id='p1',clean_gt='木門',duration=2,split='dev')]
 index=build_index(train,dev,{},setting='D')
 brief=dict(producer='offline_fixture',round_index=1,source_evidence_sha256='0'*64,summary='檢查讀音',focuses=[dict(focus_id='muk',purpose='target',level='syllable',target_unit='muk6',reason='檢查木字',evidence_ids=['step0:overall'])])
 pack=prepare_candidates(index,brief,exposures={},limit=48)
 return index,dev,pack

def test_same_patient_support_and_cross_sentence_syllable_preserve_pool():
 index,dev,pack=fixture();original=copy.deepcopy((index,pack))
 out=annotate_ucg_candidates(pack,index=index,feedback_rows=dev,current_predictions=[dict(utt_id='dev1',prediction='北門')],previous_predictions=[dict(utt_id='dev1',prediction='木門')],checkpoint_step=29)
 assert (index,pack)==original
 a=out['focuses'][0];old=pack['focuses'][0]
 assert [r['recording_id'] for r in a['candidates']]==[r['recording_id'] for r in old['candidates']]
 assert a['eligible_ids']==old['eligible_ids']
 by={r['patient_id']:r['ucg_match'] for r in a['candidates']}
 assert by['p1']['same_patient_shared_target_syllables']==['muk6']
 assert by['p2']['patient_dev_evidence_id'] is None
 assert by['p2']['shared_dev_target_syllables']==['muk6']
 evidence=a['dev_patient_evidence']['step29:DEV001']
 assert evidence['current']['target_differences']==1 and evidence['previous']['target_differences']==0
 request=build_selector_request(out)
 assert '木門' not in json.dumps(request,ensure_ascii=False) and '/dev.wav' not in json.dumps(request)
 choice=dict(context_sha256=digest(selector_context(out)),choices=[dict(focus_id='muk',recording_ids=['p1'],reason='同患者的跨句音节练习')])
 assert validate_selection(index,out,choice)['unique_recordings']==1

@pytest.mark.parametrize('mutation',['text_overlap','patient_missing','prediction_identity'])
def test_heldout_separation_and_prediction_identity(mutation):
 index,dev,pack=fixture();pred=[dict(utt_id='dev1',prediction='北門')]
 if mutation=='text_overlap':dev[0]['clean_gt']='木頭'
 elif mutation=='patient_missing':dev[0]['speaker_id']='unknown'
 else:pred[0]['utt_id']='wrong'
 with pytest.raises(ValueError):annotate_ucg_candidates(pack,index=index,feedback_rows=dev,current_predictions=pred,checkpoint_step=0)
