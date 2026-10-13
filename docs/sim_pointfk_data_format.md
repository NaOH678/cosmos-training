# 仿真PointFlow/FK数据目录规范

2026-10-09建立。所有任务以06/07的episode导出方式为准，训练前数据与训练缓存分别保存。复用通用准备器、实验配置和启动器。

## 1. 训练前数据

```text
/data/shichaojian/sim_data/bench2dex_task<TASK>/
├── data_manifest.json
├── batch_status.json
├── logs/                              # 可选
└── episode_000000[_1]/
    ├── pointflow/
    │   ├── frame_indices.npy / frame_offsets.npy
    │   ├── obs_pos.npy / obs_uv.npy / obs_track.npy
    │   ├── obs_valid.npy / obs_unique.npy
    │   ├── intrinsics.npy / timestamps_sec.npy
    │   ├── track_birth.npy / track_label.npy
    │   ├── frames/ / masks/
    │   └── recipe.json / report.json / validation.json
    ├── fk21.npz / fk21.json
    ├── state_action.npz
    ├── rgbd.hdf5
    ├── cameras.yaml
    ├── window_plan.json
    ├── complete.json / depth_audit.json
    └── head.mp4
```

预览图片、`pointflow_full.mp4`可选。保留`_1`后缀，它表示同轨迹的另一个视觉版本。导出保留原始动作与米制位移，训练前不归一化、不指定训练/验证划分。

| 数据 | 约定 |
| --- | --- |
| PointFlow | 完整episode、持续track ID；offsets `[T+1]`；位置float32 `[M,3]`、UV float32 `[M,2]`、track ID int64 `[M]`、valid/unique bool `[M]` |
| 坐标与内参 | 固定光学相机坐标、米制；PF的K按宽高归一化，FK的K为像素单位，比较前还原 |
| FK | positions_camera float32 `[T,2,21,3]`，先左后右；各机器人使用自身映射 |
| 动作 | state/action `[R,D]`，action_valid `[R]`，joint_names `[D]`，source_frame_ids `[R]`，timestamp_sec `[R]`；R为去掉回零尾帧后的帧数，T为整段源帧数 |
| RGBD | 原始RGB及对应补渲染深度，保留episode的相机、场景、seed和源帧身份 |
| 窗口 | window_plan的by_episode保留已有latent起点，每窗33帧；不另行枚举无法与latent对应的窗口 |
| 溯源 | SHA256注明对应原始下载文件还是转换后的文件；区分历史审计与本次重新验证 |

当前目录为`bench2dex_task21_v3`、`bench2dex_task27_ball`、`bench2dex_task06_fruit`、`bench2dex_task07_citrus`。27本来已符合此布局，06/07沿用本地导出。21原来的episode根级PF文件迁到`pointflow/`，根级旧名称保留兼容链接。

**21历史归档例外：** 云端只有无深度的`observations.hdf5`，统一episode目录仍使用此名称，在data_manifest及depth_audit中标记`depth_available=false`，不得改名冒充RGBD。FK/视频链接已有归档；state_action从已有动作缓存无归一化地打包；FK元数据只记录归档，不声称重新通过URDF/USD审计。完整RGBD的本地源路径在`pointflow/recipe.json`中，补档后才可解除缺项。已有训练缓存继续使用。

云端同一文件系统内允许软链接复用大文件；传桶或迁移机器时须传链接指向的实际文件，或携带全部依赖目录。data_manifest记录布局角色及缺项，不代替原生成日志。

生成06/07的本地项目记录为`/home/fangkaipeng/dexmanip_scj/Bench2Dex-Pointflow-fk`。日志可追溯`tools/run_sim_multirobot_batch.py`、`tools/generate_sim_episode_pointflow.py`、`tools/export_robot_fk_episode.py`、`tools/validate_sim_episode_pointflow.py`；当前云端未归档这些新版生成源码。

## 2. 训练缓存

```text
/data/shichaojian/sim_data/bench2dex_task<TASK>_training/
├── normalization_train_only.json
├── logs/prepare.log                   # 正式缓存生成日志
└── datasets/cosmos-cache/
    ├── video_frames -> 已有云端RGB缓存
    ├── window_index.json
    ├── vae_window_latents/window_manifest.json
    └── pointflow_fk_handguided1024/
        ├── manifest.json
        ├── episode_*.json
        └── episode_*_<start_frame>.pt
```

`datasets/cosmos-cache`为任务无关名称。21/27旧目录迁移后保留`datasets/bench2dex-task21-cosmos-cache -> cosmos-cache`链接；读取器、启动器优先新名称，也兼容旧目录。新准备目录只生成新名称。

- WAM/nonlocal/local共享窗口、latent、划分和统计；WAM不向模型提供point/FK。
- 50对轨迹按组隔离，默认seed42为94训练/6验证；已有21/27保留各自原划分。
- action统计只取训练episode有效帧，每帧一次，逐维q01/q99。offset=(q01+q99)/2，scale=max((q99-q01)/2,0.05)，state共用action统计，不clip。PT保留原始动作，读取时归一化。
- 每窗口当前帧FK引导FPS选1024点，计算RGB/法线/2cm体素特征；按track ID构建32步PointFlow位移，FK打包42点32步位移。选点不使用未来帧。
- PointFlow/FK scale分别从训练窗口有效位移计算，写入选点manifest；验证数据不参与统计。
- window_index的SHA256同时匹配选点与latent manifest；latent记录原文件引用、起点及33个源帧号。
- source_geometry_root指向训练前目录，读取器据此纠正样本中的历史预览路径。

目录整理本身不重算PT、VAE、scale、归一化或划分。06/07已另行完成全量缓存准备，复用已有latent。

## 3. 通用准备命令

CPU多进程，不占GPU，不重算跟踪、URDF FK或VAE。FK打包工具已移入`cosmos_framework/data/fk_window.py`，直接读取episode导出时无需task21的`--raw-bundle`。

```bash
LD_LIBRARY_PATH='' \
PYTHONPATH=/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python -u \
  /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/tools/prepare_sim_pointfk_v3.py \
  --local-export /data/shichaojian/sim_data/bench2dex_task06_fruit \
  --cloud-cache /data/shichaojian/raw_data/bench2dex/task06_fruit/cosmos-cache \
  --output /data/shichaojian/sim_data/bench2dex_task06_fruit_training \
  --seed 42 --workers 16

LD_LIBRARY_PATH='' \
PYTHONPATH=/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python -u \
  /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/tools/prepare_sim_pointfk_v3.py \
  --local-export /data/shichaojian/sim_data/bench2dex_task07_citrus \
  --cloud-cache /data/shichaojian/raw_data/bench2dex/task07_citrus/cosmos-cache \
  --output /data/shichaojian/sim_data/bench2dex_task07_citrus_training \
  --seed 42 --workers 16
```

## 4. 当前状态

已完成四任务各100条episode入口核对、21布局整理、四任务data_manifest写入，以及21/27训练目录迁移。旧PF路径和旧训练缓存路径保留兼容链接。

验证结果：21/27迁移前后的WAM、PointFK训练样本一致，两任务33帧验证样本读取通过；原窗口索引、归一化统计、latent清单及抽样PT哈希一致。FK打包工具与历史工具在四任务首/中/末窗口输出一致。迁移及测试临时产物已清理。

2026-10-09，06/07正式缓存已全部生成，均为94训练/6验证、1024点和42点FK。动作统计只使用各自训练集有效帧：06为43,658帧、36维，07为45,336帧、48维；state共用action统计，训练读取时补齐到64维。

| 任务 | 训练窗口 | 验证窗口 | PointFlow scale | FK scale |
| --- | ---: | ---: | ---: | ---: |
| 06 fruit | 40,650 | 2,648 | 0.0527426253548828 | 0.08296440599561948 |
| 07 citrus | 42,328 | 2,636 | 0.03377965038390645 | 0.06174585856729567 |

正式目录分别为`/data/shichaojian/sim_data/bench2dex_task06_fruit_training`和`/data/shichaojian/sim_data/bench2dex_task07_citrus_training`。每任务全部100条episode完成，索引/统计哈希一致，PT文件数匹配latent窗口数。各任务训练/验证集分别抽查首/中/末窗口，WAM与PointFK样本读取通过，共享的RGB/action/latent一致，几何标签有效值有限，点数为1024/FK42。scale保存在选点manifest，启动器自动读取。处理日志为各自`logs/prepare.log`，仓库未增加临时脚本或测试缓存。

21缺完整深度源及未归档的本地生成源码仍按前文记录，不能把训练缓存完成解释为生成环境已完整归档。
