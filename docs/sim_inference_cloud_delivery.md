# Bench2Dex 云端训练文件交接

本文与推理集群的 `bench2dex-cloud-handoff` skill 配套：云端整理文件并给出传输命令，本地接收、归档和清理暂存区。输入为用户指定的 task、模型模式、run/step 和已确认桶路径；输出为交接材料、简短清单、缺项及云端上传/本地下载命令。按当前阶段继续，不每次从头检索。

只交接文件，不修改训练或推理代码，不启动推理。默认由用户执行上传命令，明确要求代传时再执行。数据生成见 [SIM_DATA_PIPELINE.md](SIM_DATA_PIPELINE.md)。

## 1. 确认模型和来源

| 模式标识 | 含义 | 额外交接内容 |
| --- | --- | --- |
| `cosmos_raw` | 原生 Cosmos / WAM | 无 PF/FK 专属材料 |
| `pointfk_rope4d` | Cosmos＋PF＋FK，有四维局部几何位置编码 | PF/FK 配置、依赖及实际 RoPE/attention 设置 |
| `pointfk_no_rope4d` | Cosmos＋PF＋FK，无四维局部几何位置编码 | 同样保留 PF/FK 材料，明确四维编码关闭及实际其他位置编码/attention 设置 |

以实际 run 的配置、启动参数和已有训练记录确认模式，不能凭目录名或配置缺少字段判断。不能临时关闭有四维模型的开关，将它作为无四维训练模型交付。未说明模式时随配置确认；存在冲突才定向补查。

用户给出 run 时按该 run 整理；只给 task 且有多个候选时列出候选。指定多个 step 就逐项报告可用或缺失；未指定 step 先报告可用 checkpoint，不自动收集所有迭代。目录名的20k只代表目标，不代表保存了20000步。不同 task 不套用上一 task 的机器人、相机、关节顺序或归一化。

优先复用已经确认的来源和材料。新 task 不重新全仓库搜索，不翻查 Task21/Task27 历史、协议代码或完整资产库；已有资料不足、具体冲突或用户要求时才定向检索。

## 2. 整理五类交接材料

| 材料 | 查找与交付要求 |
| --- | --- |
| 完整 checkpoint | `<job>/checkpoints/iter_<九位步数>/model/`，保留 iter/model 层级、隐藏 `.metadata` 和全部 DCP 分片；不含优化器或其他训练状态 |
| 对应 run 的配置与统计 | 原样提供 `config.yaml`、`launch_info.yaml` 或实际命令/overrides/必要环境变量，以及实际使用的归一化统计；统一交付名可用 `normalization_train_only.json`，不重算内容 |
| 代表性 alignment 与机器人记录 | 一份 episode JSON，说明 task、robot_key、准确关节顺序、状态/动作表示和单位、原始及padding维度、相机顺序、预处理/必要标定、fps和chunk；引用NPZ时只补必要小数组 |
| 初始化 anchors | 本 task 对应的轻量 HDF5，完整保留 `meta` 组和必要属性、文件名及 `_1` 后缀。新接口需要 meta 外字段时另保留这些字段；已有且匹配的 anchors 复用 |
| 定制内容 | 只补官方库没有的定制资产、配置或必要代码差异，注明来源和用途。需要运行未提交源码时补必要差异或快照；当前云端版本不直接覆盖本地入口 |

记录实际归一化公式、state/action应用和反归一化位置。统计随run保存，同run的不同step通常共用固定统计。训练张量顺序以实际joint_names和映射为准，不用本地关节枚举顺序代替。

两种 PF/FK 模式均须保留：分支结构、encoder及冻结状态、点采样/映射、时间窗口、PointFlow/FK位移缩放（实际依赖逐帧文件时一并提供）、必要外部编码器依赖和位置编码/attention设置。本地已有一致依赖则复用。`FK_ENCODER_CHECKPOINT="1"`等哨兵是开关，不是待下载文件。

四维设置以该训练版本的字段为准，常见字段为 `POINTFLOW_FK_LOCAL_ROPE`、`POINTFLOW_FK_ATTN_IMPL`、`POINTFLOW_REFERENCE_ATTENTION`。已知 Task21 某个有四维 PF/FK run 为 `true / partition / false`，仅作已有实例，不套用其他run。纯WAM不交付Sonata、FK编码器、点云缓存或点定义。

公共机器人、物体、背景、纹理及依赖由本地按照仓库配置和HDF5引用，从 Hugging Face `Bench2Dex/Assets` 补齐。公共Tokenizer/VAE及其他已有匹配资源不重复传。云端不默认扫描整个资产库或下载完整泛化池。参考Task21指此前单卡流程，不采用后来8卡200条本地资产测试替代场景或协议；未来推理使用8卡也不改变交接内容。

## 3. 默认检查范围

正常文件只检查缺项、task/model/run/step归属，以及机器人、相机、归一化与模型模式是否匹配：

- checkpoint有隐藏`.metadata`和清单中的全部分片；数量复用保存记录或已有清单，不硬编码8份。已有记录确认的空rank占位也原样保留。
- config、启动参数、统计和代表性接口属于所选run；缺失项单独报告，不用其他模型或task材料替代。
- 轻量anchors提取完成、目标能打开且所需meta与属性已保留；不带RGB、动作轨迹或视频。

**不默认计算SHA256、全量回读、检查DCP storage extent、逐份anchors全量比对、审计历史协议/完整泛化池或加载模型。** 传输报错、文件异常、来源冲突、提取失败或用户明确要求时，才针对相关文件进一步检查。已有校验值可直接沿用，不为交接重新计算。

本地已有anchors且来源记录匹配则复用，不以缺少逐文件哈希作为必须重传的理由。归属仍有具体疑点时明确说明，保留相关材料待核对。

## 4. 桶、暂存与归档目录

桶路径必须经过用户确认。`cosmos_raw`、`pointfk_rope4d`、`pointfk_no_rope4d`分开，新交接按model/run区分，避免同task/step覆盖；沿用已确认的旧前缀，不为改名重传权重。

```text
<已确认桶前缀>/<必要时增加独立run>/
├── iter_<step>/model/                    # 一个或多个用户指定step
└── training_metadata/<run>/
    ├── README.md                        # 简短来源、模式、接口、去向、缺项
    ├── config/
    ├── normalization/
    ├── alignment/
    ├── robot/
    ├── anchors/                         # 需要补交时提供；保留文件名
    └── custom/                          # 仅存在定制差异时提供
```

清单只需相对路径、数量和大小，默认不生成SHA256清单或长审计报告。完整权重可从原始model目录直接上传，避免再拷贝一份大文件；配套文件在独立交接目录整理。

本地下载到 `/cpfs/user/shichaojian/rclone/`，保留task/model顶层目录。兼容旧版扁平结构、`iter_<step>/`直接包含分片和`iter_<step>/model/`结构，但新交接保留model层级。

本地skill负责归档到：

```text
/oss-shichaojian/outputs/sim_inference_assets/
├── shared/{dex2bench_dataset,model_assets,weights}/
└── taskN/
    ├── ckpt/<model>/iter_<step>/model/
    ├── training_metadata/<model>/<run>/{config,normalization,alignment,robot}/
    ├── anchors/
    ├── reference_benchmark/              # 仅实际交付时保留
    ├── sim_assets/                       # task定制资产
    └── provenance/                      # 简短来源与缺项
```

同模式多个run出现同名step时，用独立run目录区分，不覆盖已有权重。既有Task21 `ckpt/pointfk`不改名，只在记录中注明四维模式。

云端不直接操作本地暂存区或OSS。推理端按skill确认传输结束、归档齐全后清理已授权的暂存文件，保留rclone顶层；冲突或归属不明项暂留。公共资产由本地补齐，不无故移动其他task。

## 5. Task21 原生 Cosmos 传输示例

已确认run：`/data/shichaojian/runs/bench2dex/task21_normalized_94train_6val_20k_20261005`；job为`cosmos3_action/action_sft/action_policy_bench2dex_edge`，模式为`cosmos_raw`。桶前缀：`aliyun-4090-oss:pjlab-bjpai-manip/shichaojian/sim_ckpt/task21/cosmos_raw/`。

2026-10-11现场记录：用户指定19500、17000、13000、10000；19500和10000的model仍在，17000和13000只剩空iter目录，需恢复备份；未找到20000。以下只传可用两份，不用其他模型填补缺项。后续恢复时再补对应step，不重复从头检索。

### 云端上传

```bash
(
  set -e
  SRC=/data/shichaojian/runs/bench2dex/task21_normalized_94train_6val_20k_20261005/cosmos3_action/action_sft/action_policy_bench2dex_edge/checkpoints
  REMOTE=aliyun-4090-oss:pjlab-bjpai-manip/shichaojian/sim_ckpt/task21/cosmos_raw

  for STEP in 19500 10000; do
    printf -v ITER 'iter_%09d' "$STEP"
    test -s "$SRC/$ITER/model/.metadata"
    rclone copy "$SRC/$ITER/model" "$REMOTE/$ITER/model" --s3-no-check-bucket -P
  done

  rclone copy \
    /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/pointflow_outputs/task21_cosmos_raw_delivery_minimal_20261010/training_metadata/task21_normalized_94train_6val_20k_20261005 \
    "$REMOTE/training_metadata/task21_normalized_94train_6val_20k_20261005" \
    --exclude 'anchors/replay-generalization/**' \
    --s3-no-check-bucket -P
)
```

这个已有交接目录保留之前生成的校验清单，不要求新交接照搬或重算。本地已有anchors，因此示例跳过备用HDF5；如发现具体缺失再定向补传。配套压缩包与展开目录二选一，不重复上传。

### 本地下载到暂存区

```bash
rclone copy \
  aliyun-4090-oss:pjlab-bjpai-manip/shichaojian/sim_ckpt/task21/cosmos_raw \
  /cpfs/user/shichaojian/rclone/task21/cosmos_raw \
  --s3-no-check-bucket -P
```

本例下载已确认前缀内的内容，保留task21/cosmos_raw顶层；含多个非本次run或step的前缀时，改为逐个选定子目录下载，不能整桶拉取。

`rclone copy`复制目录内容，包括隐藏`.metadata`，不要用`model/*`遗漏隐藏文件。重复copy通常只补缺失或变化对象，不使用`sync`删除目标；不传checkpoint父目录，避免带优化器。`--s3-no-check-bucket`跳过建桶检查，不改变对象写权限。

## 6. 简短交接回复

返回实际来源与目的路径、task/模式/run/step、文件清单或数量/大小、接口要点、未解决缺项，以及“已准备 / 已上传”的状态。需要时附云端上传和本地暂存下载命令。无需附长审计报告；交接材料齐全不等于实际推理已经验证通过。
