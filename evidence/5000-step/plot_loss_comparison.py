"""Plot the complete 5000-step run from NPU TensorBoard and A100 CSV."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent
TAGS = ('train/loss', 'train/video_loss', 'train/action_loss')
COLORS = ('#2563EB', '#EA580C')


def read_csv(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    values = {int(row['iteration']): {tag: float(row[tag]) for tag in TAGS} for row in rows}
    assert len(values) == len(rows), f'Duplicate metric steps: {path}'
    return values


def trailing_mean(values, count=10):
    sums = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - count)
    return (sums[end] - sums[start]) / (end - start)


np.testing.assert_allclose(trailing_mean(np.array([1., 2., 3., 4.]), 2), [1., 1.5, 2.5, 3.5])
sources = json.loads((ROOT / 'sources.json').read_text())
a100 = read_csv(ROOT / 'a100_metrics_snapshot.csv')
npu_csv = read_csv(ROOT / 'npu_metrics_snapshot.csv')
events = EventAccumulator(str(ROOT / 'tensorboard_snapshot'), size_guidance={'scalars': 0})
events.Reload()
tb = {}
for tag in TAGS:
    records = events.Scalars(tag)
    tb[tag] = {record.step: record.value for record in records}
    assert len(tb[tag]) == len(records), f'Duplicate TensorBoard steps: {tag}'
steps = np.array(sorted(set(a100) & set(npu_csv) & set.intersection(*(set(tb[tag]) for tag in TAGS))))
assert len(steps) == 500 and int(steps[-1]) == 5000, 'Require exactly 10..5000, every 10 steps.'
assert len(a100) == len(npu_csv) == 500
assert all(len(tb[tag]) == 500 for tag in TAGS)
np.testing.assert_array_equal(steps, np.arange(10, steps[-1] + 1, 10))
end_step = int(steps[-1])
values = {}
for tag in TAGS:
    baseline = np.array([a100[int(step)][tag] for step in steps])
    candidate = np.array([tb[tag][int(step)] for step in steps])
    recorded = np.array([npu_csv[int(step)][tag] for step in steps])
    np.testing.assert_array_equal(candidate, recorded, err_msg=f'TensorBoard/CSV mismatch: {tag}')
    assert np.isfinite(baseline).all() and np.isfinite(candidate).all(), f'Nonfinite loss: {tag}'
    assert (baseline > 0).all() and (candidate > 0).all(), f'Log scale requires positive loss: {tag}'
    values[tag] = (baseline, candidate)

with (ROOT / 'aligned_losses.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['step', *(f'{tag}/{device}' for tag in TAGS for device in ('a100', '910b'))])
    for index, step in enumerate(steps):
        writer.writerow([int(step), *(float(series[index]) for tag in TAGS for series in values[tag])])

summary = {'scope': 'Complete 5000-step training-loss comparison; formal gates recorded separately in alignment.json',
           'start_step': int(steps[0]), 'end_step': end_step, 'points_per_curve': len(steps),
           'sampling_interval_steps': 10, 'smoothing': 'Trailing arithmetic mean of up to 10 observations (100 training steps); no dropped raw points',
           'npu_tensorboard_matches_csv_exactly': True, 'sources': sources, 'losses': {}}
recent = steps > end_step - 500
for tag, (baseline, candidate) in values.items():
    summary['losses'][tag] = {}
    for name, mask in [('common_prefix', np.ones(len(steps), dtype=bool)), ('last_500_steps', recent)]:
        bm, cm = float(baseline[mask].mean()), float(candidate[mask].mean())
        summary['losses'][tag][name] = {'a100_mean': bm, '910b_mean': cm,
                                      'signed_relative_mean_difference_percent': 100 * (cm - bm) / bm}
summary['snapshot_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [ROOT / 'a100_metrics_snapshot.csv', ROOT / 'npu_metrics_snapshot.csv',
                                         *sorted((ROOT / 'tensorboard_snapshot').glob('*tfevents*'))]}
(ROOT / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 13,
                     'axes.labelcolor': '#334155', 'text.color': '#0F172A', 'xtick.color': '#475569',
                     'ytick.color': '#475569', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#CBD5E1', 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.subplots_adjust(left=.065, right=.98, bottom=.16, top=.79, wspace=.25, hspace=.49)
fig.text(.065, .955, 'Cosmos 4B WAM  |  8 × A100 vs 8 × Ascend 910B2', fontsize=21, weight='bold')
fig.text(.065, .917, f'Complete training: steps 10–{end_step:,}  ·  batch 16/device  ·  global batch 128  ·  BF16', fontsize=11, color='#475569')
fig.legend(handles=[Line2D([0], [0], color=COLORS[0], lw=2.5, label='8 × A100 (native Cosmos)'),
                    Line2D([0], [0], color=COLORS[1], lw=2.5, linestyle='--', label='8 × 910B2 (LiteGen / NPU)')],
           loc='upper left', bbox_to_anchor=(.059, .90), frameon=False, ncol=2, fontsize=11)
fig.text(.065, .838, 'FULL 5000-STEP TRAINING', fontsize=10, weight='bold', color='#475569')
fig.text(.065, .457, f'LATEST 500 STEPS  ({end_step-490:,}–{end_step:,})', fontsize=10, weight='bold', color='#475569')
for column, (tag, title) in enumerate(zip(TAGS, ('Total loss', 'Video loss', 'Action loss'))):
    for row in (0, 1):
        ax = axes[row, column]
        mask = np.ones(len(steps), dtype=bool) if row == 0 else recent
        for series, color in zip(values[tag], COLORS):
            ax.plot(steps[mask], series[mask], color=color, alpha=.19, lw=.85)
            ax.plot(steps[mask], trailing_mean(series)[mask], color=color, lw=2.1,
                    linestyle='--' if color == COLORS[1] else '-')
        ax.set_title(title, loc='left', pad=10, weight='bold')
        ax.set_xlabel('Training step')
        ax.set_ylabel('Loss')
        ax.set_xlim((10 if row == 0 else end_step-500), end_step)
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.grid(axis='y', color='#E2E8F0', linewidth=.75)
        ax.set_axisbelow(True)
        if row == 0 and column in (0, 2):
            ax.set_yscale('log')
            ax.set_ylabel('Loss (log scale)')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.4g}'))
fig.text(.065, .075, 'Faint lines: raw values sampled every 10 steps. Bold lines: trailing 100-step mean (same smoothing for both runs).', fontsize=10, color='#475569')
fig.text(.065, .045, f'NPU: TensorBoard events (verified against CSV). A100: archived CSV from the same callback; original events unavailable. Snapshot: {sources["snapshot_utc"][:19]} UTC.', fontsize=9, color='#64748B')
fig.text(.065, .022, 'Complete training-loss comparison. Compile: A100 on / NPU off. Engineering loss gates are separate from downstream task quality.', fontsize=9, color='#64748B')
for extension in ('png', 'pdf', 'svg'):
    fig.savefig(ROOT / f'loss_comparison.{extension}', dpi=180)
plt.close(fig)
print(json.dumps({'end_step': end_step, 'points': len(steps), 'losses': summary['losses'], 'figure': str(ROOT / 'loss_comparison.png')}, indent=2))
