# badcase_demo_precision

Read-only CPU analysis of the expert (human teleop) demos used for task27 / task21: is the policy's grasp / placement imprecision inherited from the demos?

```
python demo_precision.py extract --task 27 --out OUT   # -> OUT/task27_episodes.jsonl
python demo_precision.py extract --task 21 --out OUT   # -> OUT/task21_episodes.jsonl
python demo_precision.py summarize --out OUT           # -> OUT/summary_auto.md (needs OUT/rollout_grasp_geom.json for the rollout line)
```

Data: task27 `/data/shichaojian/sim_data/bench2dex_task27_ball/episode_*/{rgbd.hdf5,fk21.npz}`; task21 objects/qpos from
`/data/shichaojian/raw_data/bench2dex/task21/replay-generalization/episode_*.hdf5`, FK from `/data/shichaojian/sim_data/bench2dex_task21_v3/episode_*/fk21.npz`.
`*_1` episodes are re-renders of the same trajectories and are skipped (50 unique demos per task).

Source of the metric definitions (4090 node, `/oss-shichaojian/outputs/infer/analysis/`), only paths / IO changed:

| here | rollout script |
|---|---|
| `grasp()` first-grasp xy offset, tips below centre, palm height | `golf_env/grasp_geom.py` (golf, right hand) and `badcase_30k40k/obj_extract.py` (auto hand, tennis etc.) |
| bottle stage condition, loss rule | `badcase_30k40k/obj_extract.py` |
| `place` block (tilt, base height at first condition) | `badcase_30k40k/topple.py` |
| `release` block (rule A) | `badcase_30k40k/release.py` |
| nearest-object distance | `badcase_30k40k/crowd.py` |

New here (not in the rollout scripts): `contact_events()` regrasp counting, `release_axis` (rule B), and the FK conversion (fk21.npz is in the cam_overhead optical frame; world = E_overhead @ [[0,0,1],[-1,0,0],[0,-1,0]] @ p).
Results and caveats: `/data/shichaojian/runs/demo_precision_20261015/summary.md`.
