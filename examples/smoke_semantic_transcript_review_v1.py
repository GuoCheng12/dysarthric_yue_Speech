#!/usr/bin/env python3
"""Live text-only policy checks; these fixtures are NEVER audio review receipts."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import mas_semantic_context_probe_v1 as run
from generate_reviewed_sg_semantic_probe_v1 import SemanticProbeReviewer, PROMPT


def main():
    base,root=run.base,run.ROOT
    p=base.verify();c=p.preference_loop;out=root/'review_policy_smoke';out.mkdir(exist_ok=False)
    reviewer=SemanticProbeReviewer(model=c.audio_review_model,base_url=c.audio_review_base_url,
        transcription_model=p.base_model_dir,transcription_base_model_sha256=c.review_asr_base_model_sha256,
        transcription_adapter_sha256=c.review_asr_adapter_sha256,allow_pending=True)
    odd='長凳幫阿強搬新生入禮堂。'
    cases=[
        ('odd_exact',odd,'我今日有啲攰。',odd,'pass'),
        ('odd_recognition_error',odd,'我今日有啲攰。','長凳幫阿強搬新身入禮堂。','pass'),
        ('intended_repetition','木框搬相框，相框搬木框。','我今日有啲攰。','木框搬相框，相框搬木框。','pass'),
        ('reference_leak',odd,'我今日有啲攰，想返屋企休息。',odd+'我今日有啲攰，想返屋企休息。','obvious_failure'),
        ('unrequested_long_loop',odd,'我今日有啲攰。',odd+'搬新生入禮堂。'*14,'obvious_failure'),
        ('reference_replacement',odd,'今日天氣好熱，我要買一杯凍檸茶。','今日天氣好熱，我要買一杯凍檸茶。','obvious_failure'),
    ]
    def check(case):
        name,target,reference,hyp,expected=case
        request=dict(model=reviewer.model,reasoning={'effort':'low'},max_output_tokens=1024,store=False,
            input=[dict(role='system',content=reviewer.prompt),dict(role='user',content=json.dumps(dict(
                target_text=target,reference_text=reference,asr_transcription=hyp),ensure_ascii=False))],
            text={'format':dict(type='json_schema',name='sg_transcript_review',strict=True,schema=reviewer.judgment_type.model_json_schema())})
        base.save(out/f'{name}.request.json',request)
        response=reviewer.client.responses.create(**request)
        base.save(out/f'{name}.response.json',base.serialize_response(response))
        assert response.status=='completed'
        judgment=reviewer.judgment_type.model_validate_json(response.output_text)
        return dict(case=name,expected=expected,**judgment.model_dump(),passed=judgment.verdict==expected,response_id=response.id)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(check,cases))
    result=dict(passed=all(x['passed'] for x in results),checks=results,actual_API=True,
        actual_ASR=False,actual_audio_inspection=False,training_eligible=False,
        purpose='Hand-authored three-text fixtures exercise the live reviewer prompt only; no fabricated transcription or admission receipts.',
        prompt_sha256=base.sha256_file(PROMPT),source_sha256=base.sha256_file(Path(__file__)))
    base.save(out/'result.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    assert result['passed']


if __name__=='__main__':main()
