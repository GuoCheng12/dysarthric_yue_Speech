"""RunStore integration for real-KL retention and residual edit+EOS audio DPO."""

from collections import Counter
from pathlib import Path
import sys

from . import preference_loop as loop
from .preference_contract import PreferenceDelivery, PreferenceSampleDelivery
from .retention_contract import RealKLRetention, load_retention_design, validate_retention_rows
from .store import RunStore, sha256_file, sha256_directory
from .text import text_normalize


def execution_sources():
    return [loop.REPO / path for path in [
        'examples/mas_retention_loop_v3.py',
        'examples/evaluate_residual_asr.py', 'examples/sample_residual_asr_preferences_v1.py',
        'examples/generate_reviewed_sg_recovery_v2.py', 'examples/generate_reviewed_sg_resumable.py',
        'finetune/scripts/qwen3_asr_edit_masked_kl_dpo.py',
        'finetune/scripts/qwen3_asr_edit_kl_dpo_continuation_v1.py',
        'finetune/scripts/qwen3_asr_residual_kl_dpo.py', 'finetune/scripts/qwen3_asr_residual_dpo.py',
        'finetune/scripts/qwen3_asr_kl_dpo_storage_v2.py',
        'finetune/scripts/qwen3_asr_patient_weighted_kl_dpo.py',
        'finetune/scripts/qwen3_asr_weights_only_patient_kl_dpo.py']]


def adapt_scope(text):
    text = text.replace('Preserve each target verbatim through SG, review and\nchosen construction;',
        'Preserve each target through SG and review. Training chosen strips only\nterminal sentence punctuation and retains original target metadata;')
    text = text.replace('No real\naudio is sampled for preferences or replayed in DPO. The 2117 real training\nrecordings are used only for existing evidence and TTS reference selection;',
        'No real\naudio is sampled for preference pairs. A separately declared subset of the\n2117 real training recordings supplies auxiliary full-GT-prefix symmetric KL;')
    text = text.replace('Optimize ordinary mean synthetic-audio DPO loss with weight 1.',
        'Optimize synthetic edit+EOS audio DPO with weight 1 plus the fixed auxiliary\nreal-training symmetric KL. This is an explicit mixed objective, not unchanged\nfull-sequence DPO or real-label cross entropy.')
    return text + '''
The initial R-only ASR is merged once into the frozen base; learned residual
weights remain unmerged during sampling, training and evaluation. Both DPO and
real-KL teachers stay at initial R-only throughout the loop. Primary training
row counts describe the preference table. Auxiliary real audio, teacher history,
weights and exposures are separately recorded in preference_delivery.real_kl_retention.
After actual current-ASR sampling, a MAS allocation component chooses integer
audio exposure counts within the fixed optimizer budget. Read those delivered
counts instead of treating requested copies or distinct rejections as dose.
More optimization steps have sometimes harmed real performance despite target
recovery. Use the frozen checkpoint gate; retain failed experience separately
from accepted parameters. Do not change step count or optimizer in a proposal.
'''


def training_pairs(pairs):
    transformed = []
    for pair in pairs:
        chosen = pair['chosen'].rstrip('。！？!?')
        if text_normalize(chosen) != text_normalize(pair['chosen']):
            raise ValueError('chosen punctuation transform changed normalized target content')
        transformed.append(dict(pair, chosen=chosen, metadata=dict(pair['metadata'],
            original_target_text=pair['chosen'], label_transform='strip_terminal_sentence_punctuation_v2')))
    return transformed


def validate_residual_sampling(protocol, receipt, parent_hash):
    expected = dict(initial_R_only_sha256=protocol.initial_adapter.sha256,
        initial_R_only_merged=True, current_residual_merged=False,
        learned_residual_loaded=parent_hash != protocol.initial_adapter.sha256,
        language='Cantonese', prompt='', do_sample=True, top_k=0,
        temperature=protocol.preference_loop.temperature, top_p=protocol.preference_loop.top_p,
        max_new_tokens=protocol.preference_loop.max_new_tokens, seed=protocol.lora.seed)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('current sampling composition or decoding differs from the frozen KL/DPO protocol')
    if receipt.get('base_model_sha256') != protocol.preference_loop.review_asr_base_model_sha256:
        raise ValueError('candidate and review ASR base models differ')


def exposure_rows(protocol, design, pairs, allocation):
    from .dpo_training import group_training_rows

    sys.path.insert(0, str(loop.REPO / 'finetune/scripts'))
    from qwen3_asr_patient_weighted_kl_dpo import weighted_window_type

    real, factors = validate_retention_rows(protocol, design)
    groups = dict(real=group_training_rows(real, 'sft')['real'],
        synthetic=group_training_rows(pairs, 'dpo')['synthetic'])
    windows = weighted_window_type(factors)(groups, design.real_per_step,
        protocol.preference_loop.synthetic_per_step, 1., protocol.lora.seed,
        synthetic_schedule=allocation['steps'])
    from .dpo_training import preference_key
    result = []
    for step in range(1, protocol.lora.optimizer_steps_per_round + 1):
        for row in windows.next():
            synthetic = row['source'] == 'synthetic'
            result.append(dict(step=step, source=row['source'], audio_id=row['_audio_group'],
                occurrence=row['_occurrence'], loss_weight=row['_loss_weight'] * (1. if synthetic else design.kl_weight),
                objective='dpo' if synthetic else 'real_symmetric_kl', preference_key=preference_key(row)))
    return result


def training_arguments(protocol, design, parent, directory):
    return dict(model_path=protocol.base_model_dir, policy_adapter_path=str(parent),
        reference_adapter_path=protocol.initial_adapter.path, real_train_file=design.real_manifest,
        synthetic_train_file=str(directory / 'mas_dpo/train.jsonl'),
        synthetic_allocation_file=str(directory / 'training_allocation.json'),
        real_loss_factors=design.real_loss_factors, preference_token_mask=design.preference_token_mask,
        output_dir=str(directory / 'mas_dpo/training'), max_steps=protocol.lora.optimizer_steps_per_round,
        real_per_step=design.real_per_step, synthetic_per_step=protocol.preference_loop.synthetic_per_step,
        synthetic_weight=1., kl_weight=design.kl_weight, lr=protocol.lora.learning_rate,
        beta=protocol.preference_loop.beta, seed=protocol.lora.seed,
        persist_teacher_cache=0, persist_optimizer_state=0)


def train_branch(root, number, arm):
    if arm != 'mas_dpo':
        raise ValueError('v3 integrates synthetic preferences plus auxiliary real KL; no branch is relabeled R/DPO')
    protocol = RunStore(root).protocol(); directory = loop.round_dir(root, number)
    design = load_retention_design(protocol); parent = loop.accepted_checkpoint(root, number - 1)
    pairs = loop.rows(directory / 'mas_dpo/train.jsonl')
    if not pairs:
        loop.save(directory / arm / 'no_update.json', dict(status='no_preference_pairs',
            training_executed=False, parent=str(parent), auxiliary_KL_executed=False))
        return False
    arguments = training_arguments(protocol, design, parent, directory)
    warm = sha256_directory(parent) != protocol.initial_adapter.sha256
    trainer = 'qwen3_asr_edit_kl_dpo_continuation_v1.py' if warm else 'qwen3_asr_edit_masked_kl_dpo.py'
    if warm:
        arguments['expected_frozen_base_sha256'] = design.expected_frozen_base_sha256
    allocation = loop.read(directory / 'training_allocation.json')
    expected = exposure_rows(protocol, design, pairs, allocation)
    loop.save(directory / arm / 'expected_exposures.json', expected)
    loop.run_command([sys.executable, str(loop.REPO / 'finetune/scripts' / trainer)] +
        [value for key, item in arguments.items() for value in ['--' + key, str(item)]], directory / arm / 'training.log')
    output = directory / arm / 'training'
    if loop.rows(output / 'training_exposures.jsonl') != expected:
        raise ValueError('actual mixed-objective exposure sequence differs from planned audio counts')
    for filename in ['policy_initial_runtime.json', 'reference_runtime.json']:
        if loop.read(output / filename)['merged_base_tensor_sha256'] != design.expected_frozen_base_sha256:
            raise ValueError('residual runtime did not use the frozen canonical merged R-only base')
    return True


def evaluate_branch(root, number, arm):
    protocol = RunStore(root).protocol(); directory = loop.round_dir(root, number)
    endpoint = loop.checkpoint(root, number, arm)
    if (directory / arm / 'no_update.json').exists():
        endpoint = loop.accepted_checkpoint(root, number - 1)
    command = [sys.executable, str(loop.REPO / 'examples/evaluate_residual_asr.py'),
        '--base-model', protocol.base_model_dir, '--initial-adapter', protocol.initial_adapter.path,
        '--split', protocol.feedback_split.path, '--out', str(directory / arm / 'residual_evaluation')]
    if sha256_directory(endpoint) != protocol.initial_adapter.sha256:
        command += ['--residual-adapter', str(endpoint)]
    loop.run_command(command, directory / arm / 'evaluation.log')


def training_registration(protocol, directory):
    design = load_retention_design(protocol); training = directory / 'mas_dpo/training'
    return dict(real_KL_manifest=Path(design.real_manifest), real_KL_loss_factors=Path(design.real_loss_factors),
        real_KL_teacher_metadata=training / 'real_teacher_cache.json',
        preference_token_masks=training / 'preference_token_masks.json',
        residual_execution=training / 'residual_execution.json', residual_verified=training / 'residual_verified.json',
        policy_runtime=training / 'policy_initial_runtime.json', reference_runtime=training / 'reference_runtime.json')


def delivery_from_artifacts(root, number):
    protocol = RunStore(root).protocol(); directory = loop.round_dir(root, number)
    design = load_retention_design(protocol); training = directory / 'mas_dpo/training'
    exposures = loop.rows(training / 'training_exposures.jsonl')
    occurrences = {(x['step'], x['source'], x['audio_id'], x['occurrence']) for x in exposures}
    counts = Counter(x[2] for x in occurrences if x[1] == 'synthetic')
    real_occurrences = [x for x in occurrences if x[1] == 'real']
    real, _ = validate_retention_rows(protocol, design)
    retention = RealKLRetention(teacher_asr_sha256=protocol.initial_adapter.sha256,
        manifest_file_sha256=design.real_manifest_sha256, loss_factors_file_sha256=design.real_loss_factors_sha256,
        teacher_cache_metadata_sha256=sha256_file(training / 'real_teacher_cache.json'), audio_count=len(real),
        unique_audio_exposed=len({x[2] for x in real_occurrences}), training_exposures=len(real_occurrences),
        total_loss_weight=sum(x['loss_weight'] for x in exposures if x['source'] == 'real'))
    reviews = {x['candidate_audio_sha256']: x for x in loop.rows(directory / 'generation/reviews.jsonl')}
    pairs = Counter(x['audio_id'] for x in loop.rows(directory / 'mas_dpo/train.jsonl'))
    slots = {x['sample_id']: x for x in loop.read(directory / 'materialization_sidecar.json')['samples']}
    generation = {x['sample_id']: x for x in loop.read(directory / 'generation/generation_summary.json')['outputs']}
    items = []
    for row in loop.rows(directory / 'synthesis_manifest.jsonl'):
        key = row['sample_id']; generated = generation[key]
        items.append(PreferenceSampleDelivery(sample_id=key,
            review_verdict=reviews[sha256_file(Path(row['output_wav']))]['verdict'],
            pair_count=pairs[key], training_exposures=counts[key], role=slots[key]['role'],
            reference_id=generated['generation_choice']['reference_id'],
            generation_mode=generated['generation_mode'], instruction_id=generated['effective_instruction_id']))
    return PreferenceDelivery(current_asr_sha256=sha256_directory(loop.accepted_checkpoint(root, number - 1)),
        frozen_reference_asr_sha256=protocol.initial_adapter.sha256,
        reviews_sha256=sha256_file(directory / 'generation/reviews.jsonl'),
        candidates_sha256=sha256_file(directory / 'sampling/candidates.jsonl'),
        pair_manifest_sha256=sha256_file(directory / 'mas_dpo/train.jsonl'),
        exposure_log_sha256=sha256_file(training / 'training_exposures.jsonl'),
        samples_per_audio=protocol.preference_loop.samples_per_audio,
        real_audio_considered=0, real_pair_audio_count=0, real_pair_count=0, real_training_exposures=0,
        synthetic_samples=items, preference_token_mask='edit_eos',
        chosen_text_transform=design.chosen_text_transform, real_kl_retention=retention)


def validate_training_artifacts(protocol, paths, training):
    from .preference_allocation_agent import ExposureDecision, compile_exposure_allocation

    required = {'real_KL_manifest', 'real_KL_loss_factors', 'real_KL_teacher_metadata',
        'preference_token_masks', 'preference_allocation', 'residual_execution', 'residual_verified',
        'policy_runtime', 'reference_runtime', 'preference_training_protocol', 'preference_training_result',
        'reference_log_probs', 'preference_exposures'}
    if not required <= paths.keys():
        raise ValueError('KL/DPO training lacks its actual auxiliary inputs and execution receipts')
    design = load_retention_design(protocol); real, _ = validate_retention_rows(protocol, design)
    run = loop.read(paths['preference_training_protocol']); result = loop.read(paths['preference_training_result'])
    cache = loop.read(paths['reference_log_probs']); delivery = training.preference_delivery
    args = run['arguments']; pairs = loop.rows(paths['train_manifest'])
    expected = dict(model_path=protocol.base_model_dir, reference_adapter_path=protocol.initial_adapter.path,
        max_steps=protocol.lora.optimizer_steps_per_round, real_per_step=design.real_per_step,
        synthetic_per_step=protocol.preference_loop.synthetic_per_step, synthetic_weight=1.,
        kl_weight=design.kl_weight, lr=protocol.lora.learning_rate, beta=protocol.preference_loop.beta,
        seed=protocol.lora.seed, persist_teacher_cache=0, persist_optimizer_state=0)
    if any(args.get(key) != value for key, value in expected.items()):
        raise ValueError('actual KL/DPO arguments differ from the frozen protocol')
    for field, path, digest in [('real_train_file', paths['real_KL_manifest'], design.real_manifest_sha256),
            ('synthetic_train_file', paths['train_manifest'], training.train_manifest_file_sha256),
            ('synthetic_allocation_file', paths['preference_allocation'], sha256_file(paths['preference_allocation']))]:
        if Path(args[field]).resolve() != path.resolve() or sha256_file(path) != digest:
            raise ValueError('trainer used different KL/DPO data: ' + field)
    if run['real_loss_factors_sha256'] != design.real_loss_factors_sha256 or sha256_file(paths['real_KL_loss_factors']) != design.real_loss_factors_sha256:
        raise ValueError('actual real-KL weights differ from the declared allocation')
    if run['reference_asr_sha256'] != protocol.initial_adapter.sha256 or run['policy_parent_sha256'] != delivery.current_asr_sha256:
        raise ValueError('KL/DPO teacher or current sampled parent changed')
    if result['status'] != 'complete' or not result['reference_files_unchanged'] or not cache['reference_unchanged']:
        raise ValueError('KL/DPO execution did not verify frozen reference completion')
    digest = sha256_file(paths['preference_training_protocol'])
    if result['protocol_sha256'] != digest or cache['protocol_sha256'] != digest:
        raise ValueError('KL/DPO result or reference scores belong to another execution')
    if Path(result['checkpoint']).resolve() != paths['checkpoint'].resolve():
        raise ValueError('actual training result points to another checkpoint')
    for key in ['policy_runtime', 'reference_runtime']:
        if loop.read(paths[key])['merged_base_tensor_sha256'] != design.expected_frozen_base_sha256:
            raise ValueError('R-only base composition differs from the declared model')
    residual = loop.read(paths['residual_verified'])
    if residual.get('learned_residual_continued'):
        previous = loop.read(Path(args['policy_adapter_path']).parent / 'training_result.json')
        if not residual['starting_parameters_equal_previous_endpoint'] or result['policy_trainable_sha256_before'] != previous['policy_trainable_sha256_after']:
            raise ValueError('current learned residual was not actually continued')
    elif not residual['zero_initial_residual'] or args['policy_adapter_path'] != protocol.initial_adapter.path:
        raise ValueError('fresh residual does not start from the frozen R-only model')
    if not residual.get('frozen_base_unchanged', residual.get('merged_parent_base_unchanged', False)):
        raise ValueError('frozen R-only base changed during training')
    teacher = loop.read(paths['real_KL_teacher_metadata'])
    if teacher['frozen_reference_sha256'] != protocol.initial_adapter.sha256 or set(teacher['audio']) != {x['audio_id'] for x in real}:
        raise ValueError('actual full-transcript teacher cache differs from declared real-KL pool')
    masks = loop.read(paths['preference_token_masks'])
    if masks['mode'] != 'edit_eos' or cache['preference_token_mask'] != 'edit_eos' or cache['mask_manifest_sha256'] != sha256_file(paths['preference_token_masks']):
        raise ValueError('preference scoring did not use the actual edit+EOS masks')
    for item in masks['pairs'].values():
        if item['token_edit_distance'] <= 0:
            raise ValueError('DPO pair has no actual scored token edit')
        for side in ['chosen', 'rejected']:
            kept = item[side + '_scored_indices']
            if kept != item[side + '_edit_eos_indices'] or kept[-1] != len(item[side + '_token_ids']) - 1:
                raise ValueError('actual preference score omitted EOS or used unregistered positions')
    allocation = loop.read(paths['preference_allocation'])
    slots = {x['sample_id']: x for x in loop.read(paths['materialization_sidecar'])['samples']}
    compiled = compile_exposure_allocation(ExposureDecision.model_validate(allocation['decision']), pairs,
        {key: value['role'] for key, value in slots.items()}, steps=protocol.lora.optimizer_steps_per_round,
        per_step=protocol.preference_loop.synthetic_per_step, seed=protocol.lora.seed)
    if {key: value for key, value in allocation.items() if key != 'agent_sources'} != compiled:
        raise ValueError('actual audio schedule differs from the MAS exposure decision')
    if not allocation.get('agent_sources') or any(sha256_file(Path(path)) != value for path, value in allocation['agent_sources'].items()):
        raise ValueError('MAS allocation request/response provenance changed')
    from .preference_allocation_agent import validate_allocation_sources
    validate_allocation_sources(protocol, allocation, pairs, slots)
    actual = loop.rows(paths['preference_exposures']); expected_exposures = exposure_rows(protocol, design, pairs, allocation)
    if actual != expected_exposures:
        raise ValueError('actual real/synthetic identities, order, weights or occurrence counts differ')
    for field, key in [('reviews_sha256', 'preference_reviews'), ('candidates_sha256', 'preference_candidates'),
            ('exposure_log_sha256', 'preference_exposures')]:
        if getattr(delivery, field) != sha256_file(paths[key]):
            raise ValueError('delivery does not bind its actual source receipts')
    occurrences = {(x['step'], x['source'], x['audio_id'], x['occurrence']) for x in actual}
    real_occurrences = [x for x in occurrences if x[1] == 'real']; retention = delivery.real_kl_retention
    expected_retention = RealKLRetention(teacher_asr_sha256=protocol.initial_adapter.sha256,
        manifest_file_sha256=design.real_manifest_sha256, loss_factors_file_sha256=design.real_loss_factors_sha256,
        teacher_cache_metadata_sha256=sha256_file(paths['real_KL_teacher_metadata']), audio_count=len(real),
        unique_audio_exposed=len({x[2] for x in real_occurrences}), training_exposures=len(real_occurrences),
        total_loss_weight=sum(x['loss_weight'] for x in actual if x['source'] == 'real'))
    if retention != expected_retention:
        raise ValueError('auxiliary real-KL delivery differs from executed audio and loss weights')
    counts = Counter(x[2] for x in occurrences if x[1] == 'synthetic'); pair_counts = Counter(x['audio_id'] for x in pairs)
    reviews = {x['candidate_audio_sha256']: x for x in loop.rows(paths['preference_reviews'])}
    assignments = {x['sample_id']: x for x in loop.rows(paths['synthesis_manifest'])}
    for item in delivery.synthetic_samples:
        verdict = reviews[sha256_file(Path(assignments[item.sample_id]['output_wav']))]['verdict']
        if item.pair_count != pair_counts[item.sample_id] or item.training_exposures != counts[item.sample_id] or item.review_verdict != verdict or item.role != slots[item.sample_id]['role']:
            raise ValueError('synthetic delivery differs from actual review, role, pairs or exposure')
