# Bench2Dex 80 / 07 / 06 / 27：云端latent与本地几何核对

云端可以先编码原始RGB；本地深度、mask、FK、PointFlow并行准备，不依赖VAE输出。
两端必须使用同一源文件和帧契约。**相同seed不能代替相同相机/场景记录。**

## 输入与输出

每任务下载目录为 `/data/shichaojian/raw_data/bench2dex/taskXX_name/dataset/<完整任务名>/replay-generalization/`。
编码输出为同级 `taskXX_name/cosmos-cache/`；原始HDF5保持不变。80/07/06/27分别对应
`task80_desk / task07_citrus / task06_fruit / task27_ball`。

| 任务 | HDF5机器人 | 数据记录的state/action维度 |
|---|---|---:|
| 80 | JAKA ZU7 + DexHand021 | 52 |
| 07 | UR5 + RH5DG2 | 48 |
| 06 | UR5 + RH56DFX | 36 |
| 27 | UR5 + Wuji | 以源文件joint_names核实，预期52 |

新入口不调用旧Wuji专用 `Bench2DexDataset`，不把其他手型重排成Wuji的52维。
state保持各HDF5 `robot/joint_names` 顺序，action按名字对齐；此阶段不计算归一化、不做FK、不定义训练/验证划分。
输出是视频编码及核对中间产物，不是已经接好不同手型的PointFlow/FK训练数据集。

## 本地必须核对的项目

1. **文件身份**：逐文件SHA256必须等于 `alignment/<episode>.json` 的 `source_sha256`。
   云端在生成缓存前实际计算HDF5哈希并与已下载官方LFS清单比较，保留 `_1` 后缀。
2. **场景、seed与相机**：JSON的 `metadata` 保存原始HDF5全部meta标量/数组，包括存在的
   seed/episode_seed、scene_generalization_sample/config、camera_definitions、robot_key、资产路径等。
   本地恢复这些记录值，不能仅设seed后重新随机化。相机顺序固定为
   `cam_overhead, cam_wrist_left, cam_wrist_right`。
3. **逐帧标定**：`alignment/<episode>.calibration.npz` 保存三路相机非RGB数值字段，包含
   intrinsic、完整逐帧extrinsic_world_from_cam、原数据中的鱼眼参数/有效mask等。
   腕部外参会随动作改变，不能只核对第一帧。云端直接使用记录的原始腕部RGB，不额外去畸变。
4. **有效帧和窗口**：保留 `time/sim_step < homing_start_sim_step` 的原始前缀，不删中间帧后重编号。
   `window_plan.json` 明确每episode的 `start_frames`，每窗取源HDF5行 `start..start+32`；
   仅接纳32个action均有效且对应sim_step间隔正确的窗口。20Hz、stride1，与task21规则一致。
5. **RGB一致性**：VAE只用原始HDF5 RGB，不用本地回放重新渲染的RGB。
   上方head 640×480、下方左右腕各320×240，拼图CHW 3×720×640。
   `rgb.pixel_sha256_TCHW` 是保留帧按时间连接的原始CHW字节哈希（不含NPY头）；
   `rgb.probe_frame_sha256_CHW` 给出首/中/尾帧哈希。本地使用附带的拼图源码逐值核对，
   并记录OpenCV版本；不同解码库的差异也需明确，不能直接声称位级相同。
6. **几何对齐**：本地回放补深度后，核对源帧、关节/物体状态、相机及深度与原RGB的投影；
   再做FK/PointFlow。不能用task21的Wuji URDF、base位姿、关节顺序代替其他机器人。
7. **最终关联**：PointFlow/FK按episode和源帧号切窗，对齐 `window_plan.json`。
   latent NPY的行号由 `start_frames.index(start)` 得到，不能直接拿start当行号。

## 云端编码参数与task21加速

入口：[cache_bench2dex_task_latents_8gpu.sh](../examples/cache_bench2dex_task_latents_8gpu.sh)，
数据适配：[cache_bench2dex_multitask_latents.py](../tools/cache_bench2dex_multitask_latents.py)；
共用VAE加载、CPU预取、编码及uint16保存：[cache_window_vae_latents.py](../tools/cache_window_vae_latents.py)。

- 8个GPU worker，按episode分配；每卡VAE batch1、BF16、inference_mode。
- RGB只解码、拼图一次并保存NPY；编码时mmap读取，每worker缓存128个resize/pad后的RGB帧，
  相邻stride1窗口复用；单独CPU线程预取下一窗口。沿用task21的加速策略。
- 每33帧窗口仍独立VAE编码，**不能整段编码后切latent冒充逐窗口结果**。
- 复用当前仓库 `find_closest_target_size/reflection_pad_to_target`：640×569内容、右侧reflect补71，
  `image_size=[640,640,640,569]`；输入float32 `/127.5-1` 后转BF16。
- Wan2.2 VAE权重 `/data/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth`，
  causal、encode_exact_durations=[33]、480档encode_chunk_frames=24；输出每窗 `[48,9,40,40]`，
  uint16保存BF16位模式。直接取mu及实现内置通道标准化，不额外采样或加scale。
- seed42只用于编码进程的可复现设置，**不覆盖源场景seed**。
- 不额外引入torch.compile、跨窗VAE状态复用等未在原task21缓存路径使用的优化。
- 每集原子发布，完成集可跳过；中断时当前未完成集会重算。复用检查绑定源HDF5、RGB、state、
  episode alignment哈希、窗口、权重和数值编码版本；源码重构保留已完成episode原metadata，
  新源码按哈希归档到`provenance/source/`。数值编码规则变化须更新`encoding_revision`。
  跳过时核对记录中的输入哈希、契约及文件大小，不每次重算所有latent内容哈希。
- shell前台输出并保存 `cosmos-cache/logs/`；27未下载齐时只等待，不读取 `.incomplete` 文件。
  四条命令顺序运行，每条都会占8卡；脚本加共享锁，防止四个任务同时抢同一组卡。

## 交给本地的核对材料

```text
cosmos-cache/
├── alignment_manifest.json         # 全episode索引及哈希/原meta
├── window_plan.json                # 精确窗口起点
├── alignment/
│   ├── <episode>.json              # 原seed/scene、RGB哈希、关节名、帧数
│   └── <episode>.calibration.npz   # 相机参数、逐帧外参、源sim_step
├── provenance/                    # 使用的源码快照、编码环境版本
├── episodes/<episode>.npz          # 原量纲state/action、valid、sim_step
├── video_frames/<episode>.npy      # 原RGB拼图缓存（较大，可留云端）
└── vae_window_latents/
    ├── window_manifest.json        # 全部完成后生成；含权重/契约哈希
    ├── <episode>.json              # 完成一集即生成；含RGB/latent SHA256
    └── <episode>.npy               # 对应窗口的latent
```

先把本MD、alignment目录、alignment_manifest.json、window_plan.json和provenance发给本地核对；
latent完成后补发metadata即可。本地不必为生成PointFlow下载全部RGB NPY和latent NPY。
源HDF5校验通过只证明输入一致，不代表本地仿真回放/几何已经核对通过。

## 本次入口验证

2026-10-07：07的 `episode_000000` 已通过源HDF5 SHA256、48维关节名对齐、原RGB导出及核对材料生成；
保留564帧、531个有效窗口。GPU1实测编码前两个窗口通过，输出uint16 `[2,48,9,40,40]`，
这是独立的限量检查，临时产物已清理，不用于正式训练。
8卡全量入口已准备，尚未替用户启动四个任务；不把两窗检查表述为四任务全量验证。
