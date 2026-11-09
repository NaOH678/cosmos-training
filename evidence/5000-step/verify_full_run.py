"""Recheck the recorded 5000-step run without starting a training process."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from mrlv.compare_wam_cross_device import compare
from mrlv.compare_wam_litegen import load_recorded_metrics, load_run
from mrlv.npu.checkpoint_verify import read_small_state, verify_component

root = Path(__file__).resolve().parent
sources = json.loads((root / 'sources.json').read_text())
run = Path(sources['npu_run'])
baseline = Path(sources['baseline_run'])
assert (run / 'status.txt').read_text().strip() == 'passed'
assert int((run / 'exit_code.txt').read_text()) == 0
before, after = load_recorded_metrics(baseline, 5000), load_run(run / 'output', 5000)
alignment = compare(before, after)
alignment.update(baseline=str(baseline), candidate=str(run / 'output'),
                 baseline_checkpoint_required=False, candidate_final_checkpoint_required=True)
(root / 'alignment.json').write_text(json.dumps(alignment, indent=2) + '\n')
assert alignment['status'] == 'passed', 'Full alignment gate failed; inspect alignment.json'
original = json.loads((root / 'original_alignment.json').read_text())
assert alignment['checks'] == original['checks'], 'Recomputed checks differ from the completion-time report'

checkpoints = list((run / 'output').glob('cosmos3_action/action_sft/*/checkpoints/iter_000005000'))
assert len(checkpoints) == 1
checkpoint = checkpoints[0]
assert (checkpoint.parent / 'latest_checkpoint.txt').read_text().strip() == checkpoint.name
components = {name: verify_component(checkpoint / name, world_size=8, replicated=name == 'scheduler')
              for name in ('model', 'optim', 'scheduler', 'trainer')}
scheduler = read_small_state(checkpoint / 'scheduler')
trainer = read_small_state(checkpoint / 'trainer')
assert scheduler['last_epoch'] == trainer['iteration'] == 5000
rng_ranks = []
for rank in range(8):
    state = trainer[f'rng_state_{rank}']
    assert {'torch', 'torch_npu', 'numpy_packed_bytes', 'random_packed_bytes'} <= state.keys()
    rng_ranks.append(rank)
metadata = FileSystemReader(checkpoint / 'optim').read_metadata()
fp32_states = {}
for name in ('master', 'exp_avg', 'exp_avg_sq'):
    tensors = [value for key, value in metadata.state_dict_metadata.items()
               if key.endswith('.' + name) and isinstance(value, TensorStorageMetadata)]
    assert tensors and all(value.properties.dtype == torch.float32 for value in tensors), name
    fp32_states[name] = len(tensors)
for name, result in components.items():
    files = [p for p in (checkpoint / name).iterdir() if p.is_file()]
    result['bytes'] = sum(p.stat().st_size for p in files)
    result['metadata_sha256'] = hashlib.sha256((checkpoint / name / '.metadata').read_bytes()).hexdigest()
checkpoint_report = {'status': 'passed', 'checkpoint': str(checkpoint), 'components': components,
                     'scheduler_last_epoch': int(scheduler['last_epoch']), 'trainer_iteration': int(trainer['iteration']),
                     'rng_ranks_read': rng_ranks, 'optimizer_fp32_tensor_counts': fp32_states,
                     'total_gib': sum(c['bytes'] for c in components.values()) / 2**30,
                     'scope': 'All component metadata/chunk references and file bounds checked; scheduler/trainer loaded; full model/optimizer tensors not loaded for an update.'}
(root / 'checkpoint_verification.json').write_text(json.dumps(checkpoint_report, indent=2) + '\n')
start = datetime.fromisoformat((run / 'started_at.txt').read_text().strip())
end = datetime.fromisoformat((run / 'finished_at.txt').read_text().strip())
report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
          'started_at_utc': start.isoformat(), 'finished_at_utc': end.isoformat(),
          'wall_seconds_including_setup_and_comparison': (end-start).total_seconds(),
          'exit_code': 0, 'training_steps': 5000, 'recorded_points': len(after['rows']),
          'world_size': after['performance']['world_size'], 'alignment_status': alignment['status'],
          'alignment_check_count': len(alignment['checks']), 'checkpoint_status': checkpoint_report['status'],
          'final_metrics': after['rows'][-1]}
(root / 'completion.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'completion': report, 'checkpoint_gib': checkpoint_report['total_gib'],
                  'optimizer_fp32_tensor_counts': fp32_states}, indent=2))
