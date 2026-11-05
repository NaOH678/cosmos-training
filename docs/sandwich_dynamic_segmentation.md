# Sandwich 自动提示分割试验

这一版先验证物体遗漏与提示框质量，不训练或微调 SAM2。人工配置的是对象清单和英文提示词；每张输入图的框由 DINO 重新检测，SAM2 image predictor 据框分割。它是逐帧基线，没有视频记忆、重新识别或 URDF 去臂，不应当作为已完成的在线部署方案。

共享配置为 `examples/deployment/sandwich_segmentation.json`。训练预处理与在线调用应使用同一份配置、模型权重和 `tools/segment_sandwich_dynamic.py` 的 `DynamicSegmenter.process_rgb(rgb)`。不要复用训练画面的坐标框或 mask。实例 mask 分开保留；legacy 语义图仍按手优先合并，因此检查边界时也应查看实例 mask。DINO 分数和 SAM2 分数都不是经过人工真值验证的准确率。

对象清单覆盖手、生菜、面包、煎蛋、奶酪、盘子、托盘。检测不到则记录 missing，不退回固定框。提示词、数量上限和阈值是试验初值，需要根据原始 RGB 和分割结果核验；不能保证每个细小或遮挡目标都能被自动找到。

先在 CPU 导出关键帧：

```bash
LD_LIBRARY_PATH='' /data/shichaojian/Track4World_portable/Track4World/.venv/bin/python \
  tools/segment_sandwich_dynamic.py \
  --episode-dir /data/shichaojian/raw_data/singlerighthand_sandwich_100/episode_0013_20260731_133649 \
  --out-dir pointflow_outputs/sandwich_dynamic_rgb_audit \
  --prepare-only
```

在 CUDA GPU 节点运行同一批关键帧，输出必须选择新目录：

```bash
LD_LIBRARY_PATH='' \
PYTHONPATH=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/pylibs_sam2 \
/data/shichaojian/Track4World_portable/Track4World/.venv/bin/python \
  tools/segment_sandwich_dynamic.py \
  --episode-dir /data/shichaojian/raw_data/singlerighthand_sandwich_100/episode_0013_20260731_133649 \
  --out-dir pointflow_outputs/sandwich_dynamic_gpu_audit
```

默认处理全部帧；加 `--frames 0,238,476,715,953` 可只检查关键帧。每帧输出原始 RGB、分割预览、实例 mask 和语义 labels，report 记录检测框、漏检对象、配置哈希和本机 DINO/SAM2 耗时。逐帧检测所有提示词会增加延迟，这些耗时不能代表未来带视频记忆、周期检测版本的性能。

人工准确 mask 可另行用于评测。若采用人工纠错生成训练输入，而推理仍全自动，则不能据此宣称两端一致。后续接入视频跟踪时，应按同样帧顺序比较离线回放与在线输入，另行验证因果性、mask/RGB/深度同步、漏检和延迟。

## 第一帧检测、后续视频跟踪

`tools/segment_sandwich_video.py` 提供 `prepare`、`initialize`、`geometry`、`track`、`merge` 模式。`prepare --out <新目录>` 解码整条 episode，按逐帧关节状态渲染手与机械臂，并在第0帧执行 DINO + SAM2 初始化。操作区为原生640×480图像上的 `[220,180,640,480]`；它是允许搜索的工作区，不是单个物体固定提示框。初始化过滤大框、桌面规模的 mask、机械臂误检以及重复对象。

`track --out <目录> --worker <0..7>` 按对象索引对8取模分配，每个对象从第0帧连续向后跟踪到结尾。禁止按时间切片后独立初始化，否则会丢失跨分片记忆。加载器仅开放当前及历史帧，若预测器访问未来 RGB 会直接报错。不运行反向传播，不在后续帧调用 DINO。首帧漏掉的对象不能自动恢复。

手部 mask 仅保留当前 URDF 手部投影12像素内、且更靠近手而非机械臂的区域。非手对象去掉明确的机械臂投影；所有 mask 限于工作区，过大的非手 mask 被置为无效。合成视频使用独占像素着色、固定类别颜色，避免重叠 mask 反复混色。

跑完8个对象分工后，使用单张空闲卡运行 `track --out <目录> --benchmark`，联合跟踪全部已初始化对象的前120帧。`merge --out <目录>` 合成完整视频、labels、实例图和延迟摘要。基准剔除前5帧，计时覆盖 SAM2 跟踪及按需JPEG加载；URDF投影与mask后处理另行计时。它不含相机传输、HTTP和策略模型，也未集成真实在线服务，不能当作真机端到端延迟。

## 面包首帧初始化的两种版本

`tools/probe_sandwich_bread.py` 仅在第0帧的架子搜索区 `[470,275,565,395]` 中检测，比较单片吐司、部分遮挡吐司、面包架等文字。框由 DINO 返回，SAM2 根据框给出可见表面 mask。它不读取后续帧来修正首帧提示。

`tools/build_sandwich_bread_variants.py` 自动筛选两片不同的棕色面包候选：有效可见面积至少80像素、棕色像素比例至少0.70、与已选候选的交叠比例小于0.5。此处排除的是首帧实际手 mask 和 URDF 机械臂，而不是膨胀的手邻域，避免把紧邻手的可见面包一起删掉。颜色与搜索区是本任务配置，不能直接当作所有任务的通用规则。

- `group`：把架子候选和两片面包的可见 mask 合并成一个组 ID。
- `separate`：两片面包各用一个独立 ID，不把架子并进面包 mask。

`tools/run_sandwich_bread_variants.py --root <两版目录>` 先单卡各联合跟踪前120帧，通过后再按8卡对象分工完成整条视频；不会沿时间切片。所有对象都从第0帧向后持续跟踪，不追加 DINO。输出 `segmentation_h264.mp4`、`labels.npy`、`instances.npy` 与初始化、计时记录。首帧很小的可见区域仍可能跟丢，需要结合完整视频评估。

## Dropper 和 micropipette

`tools/segment_robot_task.py` 使用 `examples/deployment/dropper_mix_segmentation.json` 和 `examples/deployment/micropipette_segmentation.json` 定义任务对象与首帧搜索区。搜索区内的实际提示框由 DINO 检测，不把搜索区直接作为 SAM 提示框；后续 mask 由 SAM2 跟随画面变化。搜索区依赖当前相机和初始布局，改变布局后须重新验证。

Dropper 只初始化手、烧杯、水瓶、滴管和试管架＋试管。后者由首帧分别检测的架子与管子可见 mask 合并为一个跟踪 ID，不填充物体间的空隙；此任务假定它们静止，因此后续还限制在本次自动初始化的可见范围内。相机或架子移动时应取消 `static_support`。这组 mask 供前景选点使用，本身不能提供某一个管口的独立身份。

Micropipette 分别初始化左右手、移液枪、移液枪架、试管架、透明试管、红液烧杯和右侧小杯。`episode_0007_20260824_174548` 的移液枪在首帧部分可见：局部搜索配合 `blue and white pipette` 提示和手重叠比例过滤，避免把腕部线缆误当移液枪。架子保留白色材质，红架保留红色材质；移液枪保留蓝白材质。颜色门限只能抑制部分背景和线缆误分，亦可能漏掉阴影、透明或颜色变化的部位，须检查输出。

训练预处理和推理应共用以上配置、首帧自动初始化、逐帧当前关节状态过滤及材质过滤，不能把另一条 episode 的精确 mask 复制过去，也不能用未来帧纠正训练输入。当前脚本验证的是因果离线回放，真机在线接口尚未接入，不能据此宣称已经验证完整训推一致。

本次完整视频运行：`tools/run_sandwich_bread_variants.py --root pointflow_outputs/dropper_micro_requested_20261030 --variants dropper_mix micropipette`。它先执行单卡联合对象短段测试，再完成 8 卡按对象分工的整条跟踪。

## 只跟踪动态物体，复用静止 mask

`tools/segment_sandwich_video.py` 的 `dynamic_objects` 配置决定哪些初始化 ID 进入 SAM2；配置缺省时保留原来全部跟踪的行为。静止 ID 不注册到视频预测器，每帧从本条 episode 的首帧自动 mask 恢复，再做当前机械臂排除、材质过滤、动态前景扣除，以及与首帧 RGB 的差异过滤（任一通道差值超过 `static_rgb_change_max=80` 就不保留）。这是可恢复的可见性过滤：遮挡消失时可恢复模板中的区域，不能用上一帧已经扣掉遮挡的结果累积删减。

- Sandwich：持续跟踪单手、两片面包、芝士、煎蛋和菜叶，共 6 个 ID；黑盘和托盘固定。配置见 `examples/deployment/sandwich_static_dynamic_tracking.json`。
- Dropper：持续跟踪右手、滴管，共 2 个 ID；烧杯、水瓶、试管架＋试管固定。
- Micro：持续跟踪双手、移液枪和会被拿起的透明试管，共 4 个 ID；移液枪架、试管架、红液烧杯和右侧小杯固定。

模板只包含首帧可见表面，不会补全初始遮挡后的隐藏表面。相机或这些静止物体移动时不再满足假设。RGB 差异过滤是遮挡近似，光照变化也会造成误删，同色遮挡可能漏删；有同步深度后可进一步用深度比较可见性。

`tools/run_static_dynamic_segmentation.py` 在新目录复用上述三条数据各自原来的首帧自动初始化，先在同一 GPU0 上顺序测量两种模式的前 120 帧（去掉前 5 帧），再执行完整视频跟踪。`benchmark_all.json` 是全部对象基线，`benchmark.json` 是动态对象版本。两者的 `sam_times` 排除 CPU RGB 读取归一化，包含 SAM2 跟踪执行及同步；`times` 另包含 RGB 读取归一化。每种模式都在单卡联合处理该模式全部对象，不能以 8 卡离线吞吐代替推理延迟。

输出位于 `pointflow_outputs/static_dynamic_segmentation_20261031`，包含三条完整视频、实例/语义数组、完整解码与几何限制检查、延迟对比 JSON。模板模式必须在训练预处理和真机推理使用相同规则、各自本次采集的首帧初始化及当前帧过滤；当前验证仍是因果离线回放，尚未接入真机在线服务。
