"""Realize MAS role budgets as audio exposures after actual pair availability."""

from collections import Counter, defaultdict
import random


def compile_role_allocation(sidecar, pairs, steps, per_step, seed):
    roles={x['sample_id']:x['role'] for x in sidecar['samples']}
    paired={x['audio_id'] for x in pairs if x['source']=='synthetic'}
    pools=defaultdict(list)
    for audio_id in sorted(paired):pools[roles[audio_id]].append(audio_id)
    if not pools:raise ValueError('no synthetic preference audio available')
    requested=Counter(roles.values())
    total=steps*per_step
    denominator=sum(requested[r] for r in pools)
    exact={r:total*requested[r]/denominator for r in pools}
    quotas={r:int(v) for r,v in exact.items()}
    remainder=total-sum(quotas.values())
    for role in sorted(pools,key=lambda r:(-(exact[r]-quotas[r]),r))[:remainder]:quotas[role]+=1
    labels=[role for role in sorted(quotas) for _ in range(quotas[role])]
    rng=random.Random(seed+2017);rng.shuffle(labels)
    remaining={}; sequence=[]
    for role in labels:
        if not remaining.get(role):
            remaining[role]=list(pools[role]);rng.shuffle(remaining[role])
        sequence.append(remaining[role].pop())
    return dict(schema_version='role_allocation_v1',
        rule='MAS requested role proportions among roles with actual pairs; largest-remainder allocation',
        requested_copies=dict(requested), paired_audio={r:len(v) for r,v in pools.items()},
        unavailable_roles=[r for r in requested if r not in pools], role_exposures=quotas,
        steps=[sequence[i:i+per_step] for i in range(0,total,per_step)],
        total_audio_exposures=total, loss='unchanged mean audio DPO; all unique errors average within audio')
