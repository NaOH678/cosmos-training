# PointFlow 与 FK 的局部四轴位置编码

目标是在共同相机坐标系中，让视频侧 PointFlow 与动作侧 FK 的三维相对位置参与
attention 内容匹配，从而检验 video–pointflow–FK–action 桥接是否改善。
首版采用原方案 A 的成对旋转，不新增 MLP、几何 bias 或 token。

状态：设计规格和 CPU 数学原型已完成；尚未接入训练、未验证 GPU 性能。
原稿为 `/mnt/afs/mRoPE_公式原文.md:113`；本文件明确实现选择并修正频率计算。
这里的“局部”指 point/FK token 对，不是每个簇拥有独立坐标系，也不是空间邻域硬掩码。

## 1 坐标定义

几何 token 记为集合 G，包含 PointFlow cluster 与 FK 关键点的 anchor 和未来块。
每个 token 有两份位置元数据，共享同一份模型投影得到的 Q/K：

$$
p_i=(t_i,h_i,w_i),\qquad \hat p_i=(t_i,h_i,w_i,\zeta_i).
$$

相机坐标记为 $(X,Y,Z)$，其中 $Z>0$，单位米；$\zeta$ 是无量纲逆深度编码，
不要把它与相机 Z、位移归一化 scale 混用。首版候选固定为：

$$
\zeta_i=\frac{1/Z_i-\rho_0}{s_\rho},
\quad \rho_0=1\;\mathrm{m}^{-1},\quad s_\rho=1\;\mathrm{m}^{-1}.
$$

这只是固定单位换算与公共平移，不是按样本或模态拟合的归一化。
PointFlow 和 FK 必须共用参数。公共 $\rho_0$ 在 G×G 相对相位中抵消；
其余 token 对不使用深度相位，所以也不受该平移影响。
非正、非有限 Z 必须显式报错或按定义的有效性机制处理，不静默取倒数或任意 clamp。

| 字段 | PointFlow cluster | FK 关键点 |
| --- | --- | --- |
| anchor 三维位置 | 原始成员 anchor XYZ 等权均值 | FK 经已核验的 base→camera 变换 |
| h/w | 沿用成员 anchor UV 均值和现有视频仿射 | 相机系 XYZ 投影后，使用同一视频仿射 |
| 深度 | cluster XYZ 均值的 Z，再取倒数 | 关键点相机 Z，再取倒数 |
| t | 视频首 latent 的时间原点，加未来块结束时刻 | 与 PointFlow 相同时间格 |

注意：平均 UV 一般不严格等于平均 XYZ 的投影。沿用现有 cluster UV 是为了控制改动，
此时位置是簇的摘要，不是一颗严格重建的物理点。正式接入时应记录簇内深度跨度及
两种投影的差异；若改为投影 cluster XYZ，必须作为独立、显式的坐标规则变更。
当前汇总实现见 `cosmos_framework/model/generator/pointflow_geometry.py:20`。

FK 投影必须使用与 PointFlow 一致的相机模型和原始图像尺寸，再复用 resize/crop/
拼图偏移及 patch 换算。不能直接使用未经变换的相机像素，不能独立给 FK 设原点。
内参、畸变约定和外参投影的真实窗口核验仍是训练接入前检查项；深度范围扫描不替代它。

首版使用固定 anchor 空间位置：8 个未来块沿用 anchor h/w/ζ，仅 t 推进；
未来位移仍在带噪 token 内容中。不使用未来 GT 位置、掩码或轨迹来构建位置元数据。
这编码“从哪里出发”的关系，并不直接编码未来预测位置。动态位置留作后续独立问题。

## 2 通道和频率

Edge head_dim=128，共64个二维旋转对。代码采用 split-half 布局：
第 c 对是向量维度 c 与 c+64，不是相邻的 2c 与 2c+1。
原分配为 t/h/w=24/20/20，H占1:60:3，W占2:60:3，其余为T；
因此尾部60、61、62、63均属时间，不能简单对所有64对循环取模。

原频率应为：

$$
\omega_c=(10^8)^{-2c/128}=10^{-c/8}.
$$

原稿写成 $10^{-c/4}$ 是算术错误。c=33、时间差15000对应约1.125 rad，
不是原稿的约0.000084 rad；当前窗口时间差12.8时约0.000960 rad。
代码依据 `cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py:103`。

候选四轴分配为 **24/16/16/8**：

- 时间全部保留，连同频率不变。
- H中49、52、55、58，W中50、53、56、59改为深度通道。
- 深度通道按索引排序为49、50、52、53、55、56、58、59。
- 其他56对的轴归属和频率完全保留。

选择这8对，是减少几何子块中原h/w相位的改动，不声称这些内容通道没有作用。
三轴路径不改变任何通道。深度频率的首版候选是8个等比间隔值：

$$
\nu_k=0.125\left(\frac{4}{0.125}\right)^{k/7},\quad k=0,\ldots,7.
$$

频率单位为 rad/ζ-unit。这是可检查的实验候选，不是已验证的最优值；冻结后训练
与采样使用同一份配置，并随 checkpoint 记录。不要使用原文未经核算的频率。
关闭实验开关必须恢复原三轴旋转；仅将 ζ 置零并不能恢复原h/w通道，不能用来当baseline。

## 3 成对打分

令 $R_3,R_4$ 分别为以上三轴、四轴旋转；同一个 token 的两套表示共用投影参数：

$$
q_i^3=R_3(p_i)q_i,\quad k_i^3=R_3(p_i)k_i,
\qquad q_i^4=R_4(\hat p_i)q_i,\quad k_i^4=R_4(\hat p_i)k_i.
$$

$$
s_{ij}=\frac{1}{\sqrt{128}}
\begin{cases}
(q_i^4)^\top k_j^4,& i,j\in G,\\
(q_i^3)^\top k_j^3,&\text{其他合法 token 对}.
\end{cases}
$$

所有合法 keys 共用一次 softmax；原有样本隔离和 causal/full mask 保留。
文本 K 对生成 query 的 normalization、GQA 和 packed sample 边界也必须保留。
位置改变匹配方式，不保证注意力随距离单调下降，不强制 FK 骨架与手表面点重合。

复数记法若使用 q·conj(k)，相位差是 query 减 key；原文与其正相位旋转约定不一致。
原型直接使用实数旋转后点积，避免用错误符号构造参考答案。

“局部”限定的是原始 logits 的替换范围。对于几何 query，G×G分数变化会改变
softmax分母，因此它对非几何 keys 的归一化权重也可改变；这不是额外的跨模态相位。
对非几何 query，本层同样输入下其所有分数和输出不受深度参数影响。
多层传播后，其他模态内容当然可能变化，这正是桥接希望产生的影响。

## 4 精确分组计算

非几何 query 对所有合法 keys 使用原三轴路径。
几何 query 的 keys 分成非几何集合 O 与几何集合 G：前者三轴，后者四轴。
分别得到归一化输出 $o_O,o_G$ 与对数归一化常数 $L_O,L_G$，再合并：

$$
o=\alpha_O o_O+\alpha_G o_G,
\qquad (\alpha_O,\alpha_G)=\operatorname{softmax}(L_O,L_G).
$$

这与对两组 logits 一起做 softmax 完全相同，不是两个 attention 输出直接相加。
空 key 组单独处理。不能先完整计算G×G三轴，再把G×G四轴输出直接加回去。

CPU原型为易审查仍生成完整分数，**不是可用于训练的高效后端**。
后续GPU实现应直接计算分组，利用fused attention与LSE合并；现有接口
`cosmos_framework/model/attention/frontend.py:695`要求输出/LSE指针保留、目前反向只合并
两个输出。若调整现有text/full分组，必须重新验证生产后端反向，不能依靠CPU公式代替。

不新增tokens或QKV参数，但有额外旋转、gather/scatter、kernel和归一化开销。
GPU时间/显存不能从几何子块大小直接推断。比如K=30时G=9×(30+21)=459，
这是历史簇数举例而非本次实测；实际packing容量和有效簇数需分开记录。

## 5 原型验证结果

原型：`tools/pointflow_local_mrope_probe.py`。
结果：`pointflow_outputs/local_mrope_probe_20261002.json`。

```bash
LD_LIBRARY_PATH='' OMP_NUM_THREADS=1 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python \
tools/pointflow_local_mrope_probe.py --scan \
--output pointflow_outputs/local_mrope_probe_20261002.json
```

float64小张量检查通过：4个query heads / 2个KV heads的GQA、两个变长样本、
文本因果mask、空分组、关闭后恢复baseline、时间通道保持、深度敏感性、
非几何token对分数保持、公共逆深度平移不变性、输出及Q/K/V梯度等价。
输出最大误差约1.33e-15，梯度最大误差约1.55e-15。

另按文件名选取8个episode，每个取首/中/末3个可用窗口，只读取anchor几何。
扫描不是按训练split筛选，**不用于拟合训练参数或报告验证性能**，只做量级检查。
Point数据是原始选点而非stage1簇，FK是同源帧的右手21点。

| 量 | p01 | p50 | p99 |
| --- | --- | --- | --- |
| Point相机深度 m | 0.5113 | 0.6728 | 0.8263 |
| FK相机深度 m | 0.6463 | 0.8126 | 0.9178 |
| Point逆深度 1/m | 1.2102 | 1.4863 | 1.9558 |
| FK逆深度 1/m | 1.0895 | 1.2307 | 1.5472 |

两组分布包含不同物理部位，不能由其均值差判断外参偏移或配准正确。
抽样深度全部有限且为正；这不是全量数据质量保证。
此次读取GPU状态时8张A800均约95–100%利用率，因此没有抢占训练资源做基准测试。

## 6 下一步接入与实验

1. 在当前cluster stage1 + skip(1,2) + pb4底座组合已有FK实现，保持几何运动融合关闭；
   对照与实验保持相同sigma调度、loss、选点、batch、数据窗口和训练预算。
2. 实装统一FK投影和anchor坐标元数据，核验真实head画面的point/FK叠加。
3. 实装生产分组attention，比较局部四轴与三轴的forward/backward、BF16误差、吞吐和峰值显存。
   包含空Point样本、只有FK、packing多样本、GQA和真实text K normalization。
4. 对照组为“统一FK投影 + 三轴”，实验组为“同样坐标 + 局部四轴”。
   老base的FK仍填h/w=0，只作为更粗基线，不能把投影收益全部归因于z。
5. 两组的训练与eval必须使用同一位置规则；关闭开关恢复三轴检查在生产路径重做。
   评估同一次joint rollout的video、PointFlow、FK与action对应运动的一致性，兼看成本。

本文件没有证明方案有效，也未启动新训练；它固定了一个可实现、可证伪的方案A候选。
