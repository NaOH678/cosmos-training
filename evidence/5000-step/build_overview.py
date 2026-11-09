"""Summarize existing alignment gates and measured cross-device performance."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TAGS = ('train/loss', 'train/video_loss', 'train/action_loss')
NAMES = ('Total loss', 'Video loss', 'Action loss')
COLORS = ('#2563EB', '#EA580C', '#0D9488')
plot_data = json.loads((ROOT / 'comparison.json').read_text())
alignment = json.loads((ROOT / 'alignment.json').read_text())
completion = json.loads((ROOT / 'completion.json').read_text())
checkpoint = json.loads((ROOT / 'checkpoint_verification.json').read_text())
checks = {check['name']: check for check in alignment['checks']}
assert alignment['status'] == checkpoint['status'] == completion['status'] == 'passed'
assert all(check['passed'] for check in checks.values())
with (ROOT / 'aligned_losses.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
assert [int(row['step']) for row in rows] == list(range(10, 5001, 10))
summary_rows, window_rows = [], []
for tag, label in zip(TAGS, NAMES):
    a = np.array([float(row[f'{tag}/a100']) for row in rows])
    n = np.array([float(row[f'{tag}/910b']) for row in rows])
    windows = []
    for start in range(0, 500, 50):
        am, nm = float(a[start:start+50].mean()), float(n[start:start+50].mean())
        error = abs(nm-am)/am
        windows.append(error)
        window_rows.append({'metric': tag, 'window_first_training_step': start*10+1,
                            'first_recorded_step': int(rows[start]['step']),
                            'last_step': int(rows[start+49]['step']), 'a100_mean': am, '910b_mean': nm,
                            'absolute_relative_error_percent': 100*error, 'gate_percent': 10.0})
    np.testing.assert_allclose(windows, checks[f'{tag}/500_step_windows']['relative_errors'], rtol=1e-10, atol=1e-14)
    summary_rows.append({'metric': tag, 'a100_mean': float(a.mean()), '910b_mean': float(n.mean()),
                         'signed_mean_difference_percent': 100*(float(n.mean())-float(a.mean()))/float(a.mean()),
                         'absolute_mean_error_percent': 100*checks[f'{tag}/mean']['relative_error'],
                         'worst_500_step_window_error_percent': 100*max(windows),
                         'a100_step_5000': float(a[-1]), '910b_step_5000': float(n[-1]),
                         'a100_last_500_mean': float(a[-50:].mean()), '910b_last_500_mean': float(n[-50:].mean())})
for name, data in [('loss_summary.csv', summary_rows), ('window_comparison.csv', window_rows)]:
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
perf_a = json.loads((ROOT / 'a100_performance.json').read_text())
perf_n = json.loads((ROOT / 'npu_performance.json').read_text())
fields = ('mean_step_seconds', 'samples_per_second', 'peak_allocated_gib', 'peak_reserved_gib')
performance_rows = [{'metric': key, 'a100': perf_a[key], '910b': perf_n[key],
                     '910b_divided_by_a100': perf_n[key]/perf_a[key]} for key in fields]
with (ROOT / 'performance_comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(performance_rows[0])); writer.writeheader(); writer.writerows(performance_rows)
summary = {'status': 'passed', 'scope': 'Complete 5000-step training-loss engineering comparison; not bitwise equivalence or downstream task evaluation',
           'completion': completion, 'all_20_alignment_checks_passed': True,
           'all_500_tensorboard_points_per_loss_match_csv_exactly': plot_data['npu_tensorboard_matches_csv_exactly'],
           'losses': summary_rows, 'performance': performance_rows,
           'performance_measurement': perf_n['measurement'], 'steady_measured_steps_each': perf_n['measured_steps'],
           'checkpoint': checkpoint, 'configuration': {'world_size': 8, 'batch_per_device': 16, 'global_batch': 128,
           'grad_accum_iter': 1, 'precision': 'bfloat16', 'seed': 42, 'a100_compile': True, '910b_compile': False,
           'other_numerical_settings_match': checks['numerical_config_except_compile_enabled']['passed']},
           'limitations': ['A100 uses archived CSV from the same callback; NPU uses TensorBoard with exact CSV cross-check.',
                          'Different accelerator hardware, kernels and compile policies; performance does not isolate LiteGen overhead.',
                          'DCP metadata/references and small states validated; full model/optimizer reload and another update were not run.']}
(ROOT / 'full_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#CBD5E1','text.color':'#0F172A','axes.labelcolor':'#475569',
                     'xtick.color':'#475569','ytick.color':'#475569'})
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
fig.subplots_adjust(left=.07, right=.97, top=.79, bottom=.15, wspace=.38, hspace=.58)
fig.text(.07,.95,'5000-step comparison  |  Loss gates passed',fontsize=22,weight='bold')
fig.text(.07,.905,'8 × A100 vs 8 × Ascend 910B2  ·  500 loss records/run  ·  20/20 checks passed  ·  Final DCP verified',fontsize=11,color='#475569')
fig.text(.07,.865,'Error plots use unsmoothed recorded losses. Performance is measured separately and is not a pass/fail speed gate.',fontsize=10,color='#64748B')
ax=axes[0,0]
mean_errors=[row['absolute_mean_error_percent'] for row in summary_rows]
bars=ax.bar(['Total','Video','Action'],mean_errors,color=COLORS,width=.58)
ax.bar_label(bars,labels=[f'{v:.3f}%' for v in mean_errors],padding=4,fontsize=10)
ax.set_title('Full-run mean error  |  gate ≤ 5%',loc='left',fontsize=12,weight='bold')
ax.set_ylabel('Absolute relative error (%)');ax.set_ylim(0,max(mean_errors)*1.35)
ax=axes[0,1]
for tag,label,color in zip(TAGS,NAMES,COLORS):
    error=np.array(checks[f'{tag}/500_step_windows']['relative_errors'])*100
    ax.plot(range(500,5001,500),error,color=color,lw=1.8,marker='o',markersize=3,label=label)
ax.set_title('500-step window error  |  gate ≤ 10%',loc='left',fontsize=12,weight='bold')
ax.set_ylabel('Absolute relative error (%)');ax.set_xlabel('Window end step')
ax.set_xticks([1000,2000,3000,4000,5000]);ax.set_ylim(bottom=0);ax.legend(frameon=False,fontsize=8)
for ax,key,title,unit in [
    (axes[0,2],'mean_step_seconds','Mean step time','Seconds / step'),
    (axes[1,0],'samples_per_second','Training throughput','Samples / second'),
    (axes[1,1],'peak_allocated_gib','Framework peak allocated','GiB (maximum across ranks)'),
    (axes[1,2],'peak_reserved_gib','Framework peak reserved','GiB (maximum across ranks)')]:
    vals=[perf_a[key],perf_n[key]]
    bars=ax.bar(['8 × A100','8 × 910B2'],vals,color=COLORS[:2],width=.55)
    ax.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=4,fontsize=11)
    ax.set_ylim(0,max(vals)*1.28);ax.set_ylabel(unit);ax.set_title(title,loc='left',fontsize=12,weight='bold')
for ax in axes.flat:
    ax.grid(axis='y',color='#E2E8F0',linewidth=.7);ax.set_axisbelow(True)
fig.text(.07,.08,'Performance: 4,920 steady steps per run; warmup, validation and checkpoint windows excluded; I/O and communication included.',fontsize=10,color='#475569')
fig.text(.07,.045,'Compile: A100 enabled / 910B2 disabled. Different hardware and kernels; these measurements do not isolate the effect of LiteGen.',fontsize=10,color='#64748B')
for extension in ('png','pdf','svg'):fig.savefig(ROOT/f'alignment_performance.{extension}',dpi=180,facecolor='white')
plt.close(fig)
print(json.dumps({'losses':summary_rows,'performance':performance_rows},indent=2))
