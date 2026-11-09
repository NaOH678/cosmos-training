# 本项目对 LiteGen 的实际修改

以下记录描述2026-11-06对比、2026-11-08归档的历史交付快照，基准为LiteGen提交 `ceff826ad729151fb385b1f4df787ebc4b5bd888`。当前平台已移至独立的 [LiteGen lmr-cosmos分支](https://github.com/DeepLink-org/LiteGen.dev/tree/lmr-cosmos)，四个运行时文件保留该验证快照的内容；安装不再使用项目内的LiteGen副本。跨仓库版本关系见[衔接说明](litegen_integration.md)。

## 1. 历史导出包的修改范围

57个同名文件中52个完全相同；差异仅为4个运行时文件和1个安装配置文件。`litegen/` 有51个运行时文件相同，无运行时文件新增或删除。完整开发仓库的207个示例、测试、文档、CI及其他开发文件没有进入迁移导出，这不等于删除相应运行时功能。

| 文件 | 新增 / 删除行 | 修改性质 |
|---|---:|---|
| [device/__init__.py](https://github.com/DeepLink-org/LiteGen.dev/blob/lmr-cosmos/litegen/device/__init__.py) | +195 / -4 | 应用层设备接口与同步/设备解析 |
| [litegen.py](https://github.com/DeepLink-org/LiteGen.dev/blob/lmr-cosmos/litegen/litegen.py) | +31 / -3 | 先分片、由Cosmos负责后续物化的模式 |
| [optim/master_shadow.py](https://github.com/DeepLink-org/LiteGen.dev/blob/lmr-cosmos/litegen/optim/master_shadow.py) | +6 / -5 | FP32 master初始化时序及空状态恢复修复 |
| [optim/prepare.py](https://github.com/DeepLink-org/LiteGen.dev/blob/lmr-cosmos/litegen/optim/prepare.py) | +1 / -1 | 跳过未初始化空状态 |
| [历史pyproject.toml差异](../evidence/litegen-diff/runtime-and-packaging.patch) | +8 / -145 | NPU交付依赖元数据 |

运行时代码共净增220行，含安装配置净增83行；以上不把未导出的开发资料算作代码删除量。完整逐文件结果见 [comparison.json](../evidence/litegen-diff/comparison.json)，完整差异见 [runtime-and-packaging.patch](../evidence/litegen-diff/runtime-and-packaging.patch)。

## 2. 原 LiteGen 已有的能力

`NpuBackend`、BF16支持检查、HCCL信息、RuntimeConfig通信后端校验、FSDP2/HSDP、register外部状态、register_fsdp_forward_methods及异常清理、FP32 master AdamW主体、DTensor状态校验和恢复精度保护，本来就存在。此次接入复用了它们，不应把“使用某项能力”写成“新增某项能力”。

LiteGen自身attention、checkpoint和cache运行时没有在这次目录差异中改动；Cosmos新增NPU attention位于 **Cosmos目录**，不要混淆归属。

## 3. 设备接口

新增22个函数，修改2个已有函数；底层各DeviceBackend实现未变。

| 类别 | 新增接口 / 行为 |
|---|---|
| 设备解析与选择 | device_type、get_device、is_available、is_accelerator、accelerator_available、current_device_index、set_device、empty_cache、get_compute_capability |
| stream | default_stream_context；非加速设备为空上下文 |
| 精度 | get_precision_state、set_precision、precision_context；上下文退出恢复状态 |
| 随机状态 | get_rng_state、set_rng_state、manual_seed_all |
| 标记 | range_push、range_pop、profiling_range；CUDA用NVTX，其他后端为空操作 |
| 监控 | cpu_affinity、telemetry、statistics_dtype |

选择优先级：显式参数 → `LITEGEN_DEVICE` → `COSMOS_DEVICE` → 默认加速设备。显式设备编号保留，NPU先注册TorchNPU。原 `synchronize()` 改为 `synchronize(device=None)`，按所选设备模块同步；`memory_module()` 也经过统一设备解析。

精度接口分别保留CUDA matmul/cuDNN TF32及deterministic/benchmark，NPU则区分matmul/convolution HF32。RNG接口不转换CUDA与NPU状态，不保证跨硬件随机序列相同。statistics_dtype适配后端统计/归约精度，不改变全模型BF16训练策略。

## 4. meta模型的物化职责

```python
# 新参数默认为True，保留原调用行为
runtime.prepare(model, materialize_meta=False)
```

`LiteGen.prepare()`、`_prepare_model()` 新增此模式，跳过提前的 `to_empty()`、`reset_parameters()` 和 `.to(device)`，仅对meta模型完成分片包装。Cosmos继续负责原生模型分配、初始化和DCP加载，避免框架重复处理或改变顺序。

必要检查：仅FSDP2/HSDP；本次prepare只传模型；LiteGen自身EMA、初始化/恢复checkpoint及自动恢复关闭；模型所有参数和buffer均在meta且存在可检查张量。模型物化后参数身份可能改变，后续外部优化器仍使用已有register登记。

这不是新实现一套模型构造或checkpoint加载逻辑；没有改register或特殊forward钩子的主体。

## 5. FP32 master初始化修复

构造optimizer时不再立即从模型复制master。首次step中某参数有梯度且状态为空时，才从**已加载DCP后的当前参数**创建FP32 master/moments。load_state_dict只恢复非空状态，空状态保持延迟。

原问题发生在“optimizer构造早于模型加载”的原生Cosmos顺序：随机初始化的master会在第一次step覆盖已加载的权重，即使lr=0也会发生。新行为对齐原生FusedAdam延迟初始化时序，保留已有FP32状态的无低精度中转恢复。

配套的 `prepare_fsdp2_optimizer()` 条件从 `source_state is None` 改为 `not source_state`，使 `{}` 也被跳过，不意外提前创建master。AdamW数学公式和参数组超参数未重写。

## 6. 安装配置

**以下是历史精简包的选择，不应用到当前平台分支。** Python范围固定为3.12；PyYAML范围调整为`>=6,<7`；移除CUDA/NPU/dev等extras和uv源配置、开发工具配置，由外层146包锁文件统一管理。保留setuptools核心安装，启用namespaces发现。

当前 `lmr-cosmos` 平台分支保留完整安装元数据、开发配置和其他硬件依赖组；仅将PyYAML下界从6.0.1放宽到已验证的6.0，并同步uv.lock的需求元数据。将运行时修复合入完整LiteGen开发仓库时，不必同步删除其示例、测试、文档或多设备开发配置。直接换回未打补丁的LiteGen会缺少应用设备接口、拒绝`materialize_meta`参数，并保留旧master时序问题。

## 7. 验证与证据

- [函数级差异](../evidence/litegen-diff/function-changes.json)。
- [修复审查和回归证据](../evidence/optimizer-fix/review_summary.json)。
- [两步完整硬件验收](../evidence/two-step-20261105/status.json)。
- [5000步完整对齐](../evidence/5000-step/full_comparison.json)。
- [新交付路径复测](delivery_validation.md)。

静态差异、两步功能验收、完整5000步数值对齐是三个不同证据层次。已验证范围是当前task27纯WAM，不推断其他模型、多机或可选分支的兼容性。
