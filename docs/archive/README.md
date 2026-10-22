# 归档文档（2026-10-21）

9 月到 10 月初 PointFlow/FK 开发阶段的设计、排查与实验记录，以及已被取代的交接文档。保留作历史查阅，内容可能已过时：结构、数据和结论以 docs/ 下的现行文档和实验台账 sim_task27_improvement_experiments_20261013.md 为准。注意：10-19 之前所有带 PF/FK 的训练，分支都没有跨卡同步。

| 文档 | 标题 |
|---|---|
| pointflow_action_normalization_comparison_20261005.md | Action 归一化：两套仓库实现对照（2026-10-05） |
| pointflow_alignment_audit_20260913.md | PointFlow 对齐检查报告(位置编码 / 跨模态桥接) |
| pointflow_bugfix_log_20260912.md | PointFlow 数据链路修复记录（2026-09-11 ~ 09-12） |
| pointflow_cluster_decode_20260929.md | 簇 token + 点级 decode:方案与实现记录(2026-09-29) |
| pointflow_dense_fullseq_audit.md | 新 dense full-sequence PointFlow 数据检查 |
| pointflow_displacement_scale_20260914.md | PointFlow 根因:`pointflow_displacement_scale` 没归一化(2026-09-14) |
| pointflow_eval_visualization.md | Fixed PointFlow fitting evaluation |
| pointflow_experiment_queue_20260930.md | PointFlow 实验队列(2026-09-30 建立,持续更新) |
| pointflow_fit005_analysis_20260914.md | pointflow_fit005 训练分析(2026-09-14) |
| pointflow_fk_action_coupling_20261012.md | PointFK与action耦合分析（2026-10-12） |
| pointflow_fk_local_mrope_20261002.md | PointFlow 与 FK 的局部四轴位置编码 |
| pointflow_fk_qkv_matrix_20261002.md | 当前 local RoPE 的 Q/K/V 矩阵 |
| pointflow_form_comparison_observations_20261001.md | PointFlow 形态对比观测档案(2026-10-01) |
| pointflow_geometry_motion_encoding_20261001.md | PointFlow：独立运动编码与逐点几何—运动联合编码 |
| pointflow_motion_selection_20260914.md | PointFlow 运动选点(GT 排序取 top-X% 点) |
| pointflow_per_point_tokens_20260919.md | PointFlow 逐点 token(one point = one token)拟合验证 — 2026-09-19 |
| pointflow_position_encoding.md | Cosmos PointFlow 位置编码方案 |
| pointflow_ptv3_cosmos_design.md | 基于现有代码的 PTv3–Cosmos PointFlow 接入设计 |
| pointflow_selection_visualization_20260920.md | PointFlow 选点可视化(与训练逐比特一致)— 2026-09-20 |
| pointflow_sim_v2_data_20261004.md | Bench2Dex task21 v2：数据与训练记录（更新至2026-10-05） |
| pointflow_sonata_smoke.md | 真实 PointFlow 窗口与 Sonata 编码验证 |
| pointflow_task1_data_interface.md | 任务 1：训练窗口数据接口 |
| pointflow_task2_geometry.md | 任务 2：Sonata 几何编码接口 |
| pointflow_task3_codec.md | 任务 3：Point token 与逐点解码器 |
| pointflow_task4_sequence_attention.md | 任务 4：Point token 接入 Cosmos 序列与位置 attention |
| pointflow_task5_source.md | 任务 5：真实数据源与 Cosmos 样本对齐 |
| pointflow_task6_batch.md | 任务 6：变长训练 batch 与数据容器 |
| pointflow_task7_network.md | 任务 7：PointFlow 接入 Cosmos 主网络 |
| pointflow_task8_training.md | 任务 8：PointFlow RF 加噪、监督与采样约定 |
| pointflow_task11_single_gpu_train_eval.md | 任务 11：单卡 PointFlow 训练与 eval 可视化 |
| pointflow_task12_optimizer_freeze_and_metrics.md | 任务 12：PointFlow 训练接线恢复、优化器冻结修复与可读指标 |
| pointflow_window_latent_cache_20260913.md | 逐窗口 VAE latent 缓存(方案 b) |
| pointworld_implementation_20260929.md | PointWorld 实现全解(代码走读笔记) |
| sonata_environment_validation.md | Sonata small 在 Cosmos 环境中的验证 |
| scr40_handoff_20261018.md | 从头训40k各组闭环结果交接（2026-10-18） |
