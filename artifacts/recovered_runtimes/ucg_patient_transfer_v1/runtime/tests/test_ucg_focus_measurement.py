import copy
import pytest
from therapist_harness.material_selection_offline import build_index,PracticeFocus,digest,matches
from therapist_harness.material_sampling_mas import measure_focus
from therapist_harness.sampling_curriculum_data import summarize_sampling_rows

def focus(level,unit):
 return PracticeFocus(focus_id='one',purpose='target',level=level,target_unit=unit,reason='检查开发集',evidence_ids=['step0:overall'])

def index():
 return build_index([dict(utt_id='train',audio='/train.wav',speaker_id='p',clean_gt='木',duration=2,split='train')],[],{})

@pytest.mark.parametrize('level,unit',[('text','北'),('syllable','bak1'),('onset','b'),('nucleus','a'),('coda','k'),('tone','1')])
def test_ucg_dev_profile_independent_of_train_index(level,unit):
 i=index();old=copy.deepcopy(i)
 rows=[dict(clean_gt='北',prediction='木',speaker_id='p',errors=1,chars=1,substitutions=1)]
 result=measure_focus(rows,focus(level,unit),i)
 assert result['cohort']['audio']==1 and result['cohort']['CER_percent']==100
 assert i==old and digest('北')[:16] not in i['profiles']
 if level!='text': assert result['target_occurrences']==1

@pytest.mark.parametrize('level,unit',[('text','木'),('syllable','muk6'),('onset','m'),('nucleus','u'),('coda','k'),('tone','6')])
def test_usg_existing_profile_matching_unchanged(level,unit):
 i=index();rows=[dict(clean_gt='木',prediction='北',speaker_id='p',errors=1,chars=1,substitutions=1)]
 f=focus(level,unit)
 old=summarize_sampling_rows([r for r in rows if matches(i['profiles'][digest(r['clean_gt'])[:16]],f)])
 assert measure_focus(rows,f,i)['cohort']==old

def test_no_dev_match_is_an_empty_cohort():
 assert measure_focus([dict(clean_gt='北',prediction='北',speaker_id='p',errors=0,chars=1)],focus('syllable','muk6'),index())['cohort']['audio']==0
