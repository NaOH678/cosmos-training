# Point/FK 架构实验：实现细则（2026-10-17）

配套[实验方案](pointfk_architecture_experiments_20261017.md)。本文是待实现的规格，不表示下面的模块、配置或测试已经存在。只核查代码和现有数据，未修改模型、未启动训练。

**10-17修订（v2）**：按实验方案§1.1证据改为在mask分支上以M3为基线；首个实现目标由D-base改为简化的E-rel（§4）；D1/D3、A/B/C的状态见各节开头。原D-base规格备份在 `/mnt/afs/experiments_20261017/doc_backup/`。

## 1. 实现基线与先后顺序

代码落点改为mask分支 `WorldAct-cosmos3-edge-droid-sft-pointflow-fk-mask`（`3788ac6`，含`joint_attn_mask`），新开分支实现；主工作区`3f6c3f3`没有mask，不能作为E-rel的落点。默认：M3 `physical_split`、原四模态WAM、当前相机坐标缓存、原选点/PTV3/cluster/skip/PB/local RoPE、原归一化和采样器、nolocal配置、种子42、不开indep。checkpoint、数据manifest、归一化文件、URDF和FK映射均记录路径与SHA；不要在实施时顺便更换这些设置。

顺序：第一批R1–R3零代码（实验方案§3.1），同时开发E-rel（§4）；E-rel通过§9验收后从头训40k（R4/R5）。D1等robotframe结果，D3不做；A/C延后，B本轮不做。E-rel只使用当前几何，不需要未来action→FK。

本工作区与modemix、cons不是一个版本。WAM/FDM/IDM混合是独立训练变量，首轮不开。M3是E-rel的默认attention设置，mask规则不改；E-rel注入在embedding层，不受mask影响。

## 2. badcase与消融如何约束设计

| 证据 | 本轮具体决定 | 不据此做什么 |
| --- | --- | --- |
| 几何预测较好，但action对几何干预较弱 | D直接增强action输入；B直接修正action输出，分别检验 | 不再只优化独立FK头的显示效果 |
| 手和球共同偏，接触窗相对误差小于绝对误差 | 保留全局相机特征；D1补掌部坐标表达；考核绝对落点 | 不把手球相对距离loss当主损失 |
| 箱子/球已有高度信息，相机高度也随机 | D1使同一物体相对手掌的位置更易读取 | 不补假桌面点，不把相机z直接当桌高 |
| task21主要问题含规划松手过高、瓶身倾斜 | 记录松手阶段动作FK高度/朝向与专家偏差 | 不用统一“降低手高度”或距离越小越好的奖励 |
| 动作干预对视频影响弱，GT视频可显著改善action | C单独建立本次action→video通路；D3只测试当前几何输入 | 不把D3称为未来动作控制视频 |
| 推理补球点600局无改善，P3提前重规划无改善 | 保持选点及execute_steps32 | 不认为结构能凭空恢复0点目标或学出未演示的重抓 |
| M3从头训30k/35k显著优于同期基线（合并p=0.006） | 新结构以M3为基线，mask不变 | 不声称已有实验支持“增加所有模态互联” |
| M3切的是整个模态对（含anchor），action看不到任何PF | E-rel只把**当前干净**的手–物关系送进action输入；R3零代码检验“放开当前PF anchor” | 不让action重新读带噪PF（B的做法） |
| M2/M1/M4：action读带噪几何无益，切action←带噪video也无益 | B本轮不做；C延后 | 不把“更多耦合”当默认方向 |
| 4k续训把结构差异压平（ft40） | 从头训40k，按30–40k合并判断 | 不用4k续训筛选或否定结构 |
| 原方案自查：首窗口10根指尖3 cm内全无物体点 | kNN不设硬半径，RBF距离编码，另给每实例最近距离 | 不用硬3 cm邻域 |

证据入口：[主实验台账](sim_task27_improvement_experiments_20261013.md)第3、3.5节及台账30k/P3行；cons的`docs/pointfk_consistency_diag_20261014.md`第4—7节。80条badcase仅采用用户给出的报告，未重新逐案观看；这些选中失败不能用于估算整体可挽救比例。

## 3. 数据接口：补一个逐点实例字段，不改选点

以下形状是当前双手、32步配置。实现应从timing和映射读取，首版仅支持该配置时明确检查，不静默套用其他机器人。

| 变量 | 形状/单位 | 来源与约束 |
| --- | --- | --- |
| 当前关节q0 | 每样本`[52]`，物理关节单位 | action第0个条件行反归一化；不是未来GT action |
| 当前FK anchor | 每样本`[42,3]`，米 | 原FK输入，左右各21点 |
| 当前点XYZ/UV | 拼接后`[N,3]` / `[N,2]` | 原PF inputs；N为batch各样本点数之和 |
| 细粒度点特征f | `[N,C0]` | `geometry.voxel_features[original_to_voxel]`；当前C0=32，代码读取实际维度 |
| cluster映射 | `[N]` | `geometry.original_to_cluster`；总cluster数K，不重新聚类 |
| 当前点类别 | `[N]`，int8 | v2：只需“任务物体实例k / 容器 / 其他”。由数据准备时的位姿几何归属得到（见下），**作为前向输入**（仿真两端都有真实值） |
| 当前点实例 | `[N]`，int8 | 0..K-1为任务物体（task27四球、task21三瓶），K为容器（task27箱子），-1为其他（含手、桌、干扰物） |
| 原PF未来监督 | `[32,N,3]` / `[32,N]` | 原位移/valid，只进原loss及指定的新loss，不用于D选邻居 |

**v2实例来源**：`tools/prepare_sim_pointfk_v3.py`在`--object-quota`时已用仿真box3d在anchor帧计算`object_membership(p, objects)`（与推理`anchor_selection.object_membership`同一规则），但**没有存进样本**。改动：
- 无论是否开quota，都计算membership并存为样本字段`pointflow.inputs.point_instance = member[keep]`（int8）；箱子作为第K类加入objects列表（kind=box）。
- 只重跑点缓存准备这一步（不重跑tracking）；VAE等视频缓存能否直接复用（软链/共享路径）实施时核对，不能复用则先问用户再决定。
- 选点保持默认`hand_guided_fps`，生成新bundle（如`bench2dex_task27_ball_training_rel`、`bench2dex_task21_v3_training_rel`）；与原bundle逐样本比对`point_ids`完全一致，只多一个字段。
- `PointFlowBatch`的inputs增加可选`point_instance`（拼接、`.to()`/`.to_cuda()`），缺失时填-1；开启E-rel而样本无该字段时报错，不静默当作“无物体”。
- 推理端：4090 `sim_geometry`里已有`_quota_camera_objects()`+`object_membership`（objq/ballquota分支），对最终选出的1024点算同样的`point_instance`传给模型。训推须用同一份objects列表（含箱子）和margin。

原方案的`track_label`读取与分类头在v2首版不需要（只用物体实例，手的状态由FK anchor提供）；真机阶段再加分类头。

原方案统计的“首窗口10根指尖3 cm内均无物体点”是v2放弃硬半径的直接依据；训练日志按阶段记录kNN最近距离分布。

## 4. E-rel：当前手–物关系 → action输入（v2，替代原D-base）

### 4.1 接入位置

不需要`prepare_geometry`重构，不读PTV3特征，不改PF/FK payload。只在`mot/cosmos3_vfm_network.py`中`_encode_action`产生action输入embedding之后、送入主干之前加一个增量：

```python
# 伪代码；禁止读取任何未来监督
rel = relation_encoder(pf.inputs.anchor_xyz, pf.inputs.normal, pf.inputs.rgb, pf.inputs.point_instance,
                       fk.inputs.anchor_xyz, pf.sample_offsets)          # [B, H]
action_emb[rows_of_sample_b] += gate * rel[b]                           # 33行全部，含条件行
# 可选（pointfk_relation_to_fk_anchor）：
fk_emb[anchor_rows_of_query_q_in_sample_b] += gate * proj_fk(site_feat[b, q])
```

样本归属用packing的payload/span元数据，不能用`global_index // 33`猜。FK anchor行用FK payload里的当前anchor索引（若无显式字段，新增`anchor_indexes`）。

### 4.2 查询点与邻点

- 查询点12个：左右手各`[0,4,8,12,16,20]`（掌心+5指尖），右手索引加21；映射与现有42点定义不符即报错。
- 候选点：`point_instance >= 0`（任务物体+容器）。每个查询点取k=16个最近候选点，**不设半径**；候选不足16时用有效mask补齐，不重复取点。
- 批内实现：FP32距离`[B,12,Nmax]` + padding mask → `topk` → gather `[B,12,16,*]`；无Python循环、无CPU kNN、无`.item()`。

### 4.3 编码

逐邻点特征（单位米）：
- `Δxyz / 0.05`（邻点−查询点，相机系）；
- `RBF(d)`：8个中心均匀分布在0–0.20 m，宽0.025 m；
- 法向（3）、RGB（3，[0,1]）、实例one-hot（K+1）、有效位。

共享MLP `→64→64`（SiLU），对16邻点做有效mask下的max与mean，得到128维。再拼每个实例的“最近距离（RBF 8维）+ `log1p(该实例在1024点中的点数)` + 无点标记”，经`Linear→128`得到查询点特征`site[B,12,128]`。

### 4.4 注入

- action：按手拼接`[B, 6×128]×2 → 1536` → `MLP(1536→512→H)` → 乘门控 → 广播加到该样本全部33行action输入embedding。数值、condition mask、loss mask不变。
- 可选FK anchor：`site[B,q] → Linear(128→H)` → 加到对应的12个当前FK anchor token（不改未来FK变量）。首个训练run（R4）开启与否需在提交前定，R5必须相同。
- R5 E-rel-abs：同一网络与参数量，只把`Δxyz`改为邻点相机系绝对坐标`xyz/0.5`、把RBF距离改为`RBF(‖xyz‖)`，其余不变。

**初始化**：每条注入的最后一层权重与bias为0，前层正常初始化，门控固定为1。开关关闭时不构造模块，不改变原模型随机初始化顺序。H取网络`hidden_size`，不写死。

## 5. D1与D3：各自只增强一处

> v2状态：D1等robotframe 25k/30k结果（robotframe有效则优先全局机器人系+E-rel）。若做D1，把§4.3的`Δxyz`旋到当前掌坐标，其余沿用E-rel。D3不做（M3下video已直接读PF）。以下保留原规格。

### D1：补手掌坐标，不改RoPE/缓存坐标系

扩展`TorchRobotFK`，新增返回指定link完整变换的方法，原`forward(q)`返回关键点的接口保留。当前实现最后已经应用`world_from_robot_base`；返回手掌变换时也必须应用它，不能关键点在world而朝向还在base。

若缓存为相机坐标，用当前`q0`计算的42点与当前FK anchor估计一次刚体`R,t`，只对齐当前状态；已提供并验证的frame transform优先复用。将掌部位置和朝向一起变换到PF所在坐标。不得拿预测未来/GT未来逐帧拟合。机器人坐标缓存另走明确分支，不能重复转换。

若`R_palm`将掌坐标列向量旋到相机坐标，点的行向量局部坐标为`(point_cam - palm_translation) @ R_palm`。D1给关系编码补左右手各自的局部XYZ及指尖距离，不替换原相机XYZ、UV或位置编码。先验证掌中心得到0、掌X轴前移1cm得到局部`[0.01,0,0]`；再验证已知整体刚体变化时局部坐标不变。

刚体拟合残差、单位和左右手映射必须记录；误差超出该数据的已核查容差时启用诊断/报错，不能悄悄将错误变换作为训练输入。这个实现只需要当前状态、当前anchor、现有URDF，不读GT未来。

相机坐标的约定一致不等于真实数据已经精确米制配准。PF继续用自己的估计内参，FK继续用自己的现有投影；拟合当前FK不能修正PF深度/尺度误差。因此本轮主要落在已核查的仿真数据，迁移真机前需重新量测相对几何误差，不能把3cm当作跨数据源的真实接触距离。

### D3：只改当前视频输入embedding

使用D的逐点两类关系特征，经已有`uv_to_video`得到当前点在拼图画布上的像素。复用`video_aligned_point_positions`的像素中心规则：转patch后含`+0.5/stride-0.5`修正；不要额外重复resize。`stride`取实际latent/patch总缩放，不写死16。

以双线性权重向相邻4个patch scatter，手/物体分两路累计`sum(weight * probability * feature)`与`sum(weight * probability)`，各自归一化，附两路coverage；投影到H后加到原vision embedding。没有覆盖的patch严格增量0，越界点丢弃，不钳到边界。聚合不采用未归一化求和，防止手点数量压过小球。

只处理每个视频payload的第0个已观测latent、主视角实际区域。使用`ModalitySpan.payload_index/payload_start`与`token_shapes`定位flatten后的行；使用真实view区域与pad信息裁剪，不能用RoPE的全局空间origin当数组索引。多样本、拼图下半部腕视角、反射padding区都不能串入。IDM即使全部视频为条件，也只注入当前第0帧。

建议在全部原模态编码完成后统一调用`frontend.inject`；它作用于`_encode_vision`产生的embedding，不必在`_encode_vision`内部提前写入随后会被覆盖的数据。D3不改VAE输入/latent、未来vision条件mask或原视频loss。梯度经特征与新投影回到前置模块；当前UV本身不需梯度。

## 6. A/B/C共用的预测动作→FK接口

> v2状态：A、C延后到第三批，B本轮不做（理由见实验方案§3表）。以下保留原规格。在M3上实现A1/C时注意：A1的FK→PF适配器、C的FK→video适配器都绕过了M3切掉的边，需作为显式设计决定，而非默认。

新增独立`action_geometry.py`，不要调用只在loss里使用的`pointfk_consistency.action_x0_hat`：后者读取训练epsilon和目标速度重建输入，不能成为推理前向依赖。

建议接口`estimate_clean_action(action_xt, predicted_velocity, sigma_per_row, condition_mask, domain_id)`。条件行原样返回`action_xt`；生成行用FP32的`action_xt - sigma * velocity`。sigma来自动作实际采样/加噪状态，不用video sigma或timestep embedding值代替。通过打包元数据映射逐行sigma，不能将一个noisy-only向量误广播到包含条件行的33行。

接着按训练使用的domain归一化参数反变换，只取有效52维并核对joint_names，再用`TorchRobotFK`计算关键点。原action padding到64的其余维不送入URDF。当前q0决定相机刚体变换，所有32步共用；未来关节角保持可微。关节越限/NaN必须统计；首版不把越限硬裁剪偷偷写回最终动作，几何支路如需稳定限幅必须独立配置并记录饱和比例。

上述52维接口仅适用于当前关节指令domain。其他数据若是末端位姿/增量控制，直接拒绝此FK路径，不能把它当作52个关节角；几何helper应有明确domain检查。

WAM/IDM用本次干净动作估计；FDM直接用完整给定动作。当前`_decode_action`在没有noisy action时返回dummy零值供图一致性使用，**不能拿它当FDM动作**。新helper应从条件行读真实给定值。所有几何分支强度默认乘`(1-sigma_action)^2`，条件动作取sigma=0；独立sigma训练时不能错用PF/video的噪声强度。

FK输出单位固定米，语义固定command FK。原数据未来FK来自实际状态，不能当相同目标混用。32个物理步先完整计算；确需8组token时每组拼4步的位置/位移后编码，不只拿一个端点代替整组。

### A：需要同时更改采样契约，不只是删loss

先将`_decode_action`移到PF decode前（只解码一次），恢复干净动作，产生FK token，再以局部适配器条件化PF decode。A0只去未来FK，A1再加适配器。当前anchor编码需与未来motion编码拆开：不能给原FK branch喂全零motion并继续附加336个“假未来token”，那不叫移除未来变量。

A1首版把每4步FK位移拼接，投影成`[8,42B,128]`；PF的`point_hidden[8,K,H]`投影为128维query，同样本/同时间组读取42个FK key/value（2个head，每head64维），再经零初始化投影残差加回原PF hidden，交给原codec decode。该小cross-attention不增加主干token，也不修改主干attention backend。加入当前点到骨段的关系描述，物体支路保留独立MLP和有效性特征，不把FK位移直接加到物体GT目标；混合cluster按两路成员权重合成。A1只测试这条解码条件通路，不同时强制手部PF改成“FK位移+残差”输出，避免额外改变监督参数化。

`OmniMoTModel`的训练噪声构造、joint state拼接/拆分、velocity field、采样积分、eval输出都需排除未来FK；anchor仍是条件。最终eval新增`fk_command`由最终action算出，原`preds_fk`/独立FK ADE不得继续伪造返回。A0/A1丢弃旧未来FK专属权重时列加载清单，EMA同样处理。

### B：最终action只有一份，修正的是速度

原主干先解码原PF与初步action；用它们的干净估计构建手/物体关系。PF位移乘原displacement scale后加anchor，不能把归一化位移当米。手部绑定使用当前固定对应；物体在各预测物理步按3cm/16点读取邻域，仅使用预测位置与当前类别，不读取GT valid。

两路摘要与该时刻action hidden拼接，MLP输出`delta_velocity[32,52]`，padding维和条件行置0。最终`v_action = v_base + (1-sigma_action)^2 * delta_velocity`，再按原接口计算action loss/采样更新。最终command FK重新由最终action估计计算。首版不把该修正再送回PF，避免同次循环。

B0用相同维度、近似参数量的普通PF特征交互，B1才分身体/物体；两个组使用相同loss。如果初步action直接辅助监督，B0/B1也必须相同；首版仅监督最终action更容易比较。

### C：可微投影必须送到实际视频输出路径

解码action/PF后恢复32步干净几何，分别用已有内参及真实画布映射soft splat到视频patch。FK与PF不互换内参；PF内参由normalized矩阵恢复时用自身`image_size_wh`。z≤0或超出主视角者无贡献，不钳边；输出中带覆盖和sigma。四个物理步各自投影后，将对应特征拼接压缩为一组，按既有timing连接到8个未来latent时间组。

首版`video_delta = zero_init_projection(MLP(concat(video_hidden, geometry_map)))`，只加到被覆盖的未来主视角hidden，再交给原`llm2vae`输出速度；分支强度用action噪声门控，PF通道再按PF噪声独立门控。无需加第二遍主干或新大型图像编码器；IDM没有生成视频时不修改条件内容。

纯FK辅助loss能更新action，不能保证video服从它；C这个前向适配器才提供同次响应。投影位置的梯度可经双线性权重和FP32 FK回到action；patch索引本身不可微，越界/空覆盖也没有该路径，所以必须在可见内点单测，不能只检查总loss有限。

## 7. Loss：先固定原损失，只逐项增加

| 组 | 首轮loss | 可选后续消融 |
| --- | --- | --- |
| R3 M3a、R4 E-rel、R5 E-rel-abs | 原四模态loss，不加任何新loss | 视结果再比较command FK loss |
| D1（若做） | 同R4 | — |
| B0/B1（本轮不做） | 原四模态loss + 同一type_CE | 单独比较手部关系loss |
| A0/A1（延后） | 原video/action/PF loss + 相同类别设置 | 单独比较command FK loss |
| C（延后） | 原四模态loss + 同一type_CE | C0只适配器；C1另加局部video loss |

（以下type_CE仅适用于真机阶段或B/C，v2首版不用。）0.01是首次试跑的分类系数建议，不是实验得出的最优值。CE按有标签当前点归一化，未知点ignore；额外记录两类召回/混淆与不同类别来源。D-match也保留相同分类辅助头/loss，但不用其输出分流，以免混入额外监督差异。

可选command FK loss：预测action经FK与**GT指令经同一FK**比较，距离先除0.03m，再用SmoothL1，按有效物理步/关键点归一化，乘动作低噪声权重；不拿实际状态FK强行作command目标。可选手部关系loss：预测PF与动作FK的相对关系，比较其GT PF与GT指令FK的相对关系，不将差值逼为0；固定当前对应，沿预测action和PF双向反传。

C1局部loss使用原视频**速度误差**，在GT当前/未来手部投影或已有物体框覆盖的patch上按有效patch归一化，建议额外系数0.25；无有效ROI该项0。ROI只在loss构造，不进入网络或选邻居。VAE latent patch不等于精确像素区域，这只是局部加权，不冒称像素级接触监督。若物体框资源尚未接好，先只用现有手部GT轨迹投影并注明不含未选中小球，不能把缺失区域写成全0负样本。

第一轮不加穿透、接触保持、物体随手、恢复奖励。所有新loss同时记录未加权值、有效数量及对action/PF/视频头的梯度量级；不能仅凭系数10:1推断谁支配训练。

## 8. 配置、改动文件与提交边界

以下为**拟新增**配置，不是可立即执行的环境变量：

```text
# v2首版
pointfk_relation = off | rel | abs          # R4=rel，R5=abs
pointfk_relation_k = 16
pointfk_relation_rbf_max_m = 0.20
pointfk_relation_to_fk_anchor = false | true
# 保留给后续（v2首版不实现）
pointfk_frontend = off | base | palm_local | video_input | palm_local_video_input
pointfk_type_source = predicted | observed
pointfk_output_adapter = off | action_to_pf | geometry_to_action | geometry_to_video
pointfk_future_fk = predicted | action_derived
```

参数进入结构化model config（与`joint_attn_mask`同处），launcher只负责传参；同时打印最终解析值。禁止仅训练读env而部署默认关闭。

| 建议提交 | 文件/职责 | 完成标准 |
| --- | --- | --- |
| 1 数据 | `tools/prepare_sim_pointfk_v3.py`存`point_instance`；`sim_pointfk_dataset.py`、`pointflow_batch.py`打包；生成`*_rel` bundle | 与原bundle `point_ids`逐样本一致；旧样本兼容；缺字段报错 |
| 2 关系模块 | 新`model/generator/pointfk_relation.py` | 批内kNN、无硬半径、空实例/不足16点不NaN |
| 3 注入 | `mot/cosmos3_vfm_network.py`（`_encode_action`之后）、模型配置 | 关闭逐位等价M3；零初始化输出等价；token数/RoPE/mask不变 |
| 4 推理端 | 4090新worktree `Bench2Dex-Pointflow-fk-rel`（基于含`object_membership`的objq/ballquota分支）：vendor提交2/3到`third_party/worldact_runtime`；`sim_geometry`/`live_geometry`产出`point_instance`；`tools/prepare_sim_inference.py`写对应`worldact_root` | 同一训练帧在训练端与推理端算出的`point_instance`和`site`特征一致 |
| 后续 | D1（`robot_fk_torch.py`掌变换）、A/B/C（`action_geometry.py`等，原规格） | — |

不把训练dataset导入推理核心；model模块只消费张量契约。旧checkpoint/EMA初始化新增参数保持相同，恢复新checkpoint时验证配置一致。

## 9. 必须通过的验收与实验记录

### 开始训练前：小而有针对性的测试

1. **数据**：manifest路径、point_ids、标签对齐；batch两样本不同点数，拼接buffer无别名；旧样本无标签时可运行。
2. **局部关系（v2）**：k近邻正确（与numpy暴力解比对）；候选不足16时有效mask正确、不重复；某实例0点时“无点标记”=1且不NaN；`point_instance=-1`（手、桌）的点不进入候选。
3. **packing**：两样本不同K、无几何样本、文本/动作跨度变化；PF/FK/action增量只落在该样本对应索引。D3额外检查主/腕视角、当前/未来、padding和越界UV。
4. **零初始化与梯度**：同seed原模型/关闭开关输出等价；开启零初始化输出在数值容差内等价；第二次优化后关系模块有梯度。无几何时新参数仍需接入零贡献计算图，避免不同rank稀疏样本触发unused parameter问题。
5. **训推一致（v2新增）**：取一个训练episode首帧，在4090用live geometry重放，比较`point_ids`、`point_instance`与`site`特征（容差1e-5）；推理日志打印E-rel开关与worldact_root。
6. **时间/噪声**：逐行动作sigma、条件行保持、52/64维、32步/8组对应；FDM全条件、IDM全视频条件、独立sigma都单独测试。将GT未来/GT类别改成随机数，应不改变固定模型同输入的前向输出，只改变相应loss。
7. **几何**：FK与现有实现同关节同点顺序比较；掌局部坐标刚体不变性；中心附近的FK/投影梯度与有限差分一致。A额外验证采样state确实没有未来FK维度。

之后非rank0单卡batch1跑一次训练前向/反向、保存恢复与完整joint eval，再8卡检查真实max batch16、compile和选择性checkpoint兼容。attention继续flash2+varlen，不引入natten，不为了新模块恢复全量稠密attention矩阵。

几何距离批内可用`[B,10,Nmax]`的FP32距离加padding mask，topk后gather到`[B,10,16,*]`；避免Python逐指尖循环、每step CPU kNN或`.item()`同步。3cm筛选发生在point层，不把1024个原始点新增为主干token。记录step时间和推理时间再决定缓存当前context，首轮不把缓存加速与架构收益混为同一实验。

### 训练后：同时验证“通路存在”和“通路有用”

- 固定观测与噪声，关闭新增模块，对比action规划的绝对落点/松手位置与视频局部变化；D不能用未来动作响应作唯一验收。
- B/A/C使用小幅可行动作干预，并在每次去噪持续注入对应动作；与相同输入重复采样噪声比较。PF/视频需在对应手部方向正确地变化，而非整幅图随便变。
- task27分别统计0点目标、少点、正常覆盖及物体类别；用原HDF物体真值作离线评估，不送入模型。ADE仅针对已选点，不能代表目标没被遗漏。
- task21重点看松手高度、姿态和放后撤离/碰倒； FK掌姿态只能反映手，瓶身倾斜仍需原物体真值或视频另测，不能等同。
- 同一训练预算、guidance1、4步、binomial5、execute_steps32，四通道同seed配对；先200局/任务作筛选，有收益后补独立训练种子与30/35/40k，不能只挑最高checkpoint。M3 30k结果仍属于待跨点确认的证据。

每组记录：代码commit、完整配置、缓存manifest/normalizer/URDF SHA、加载/跳过权重、EMA、分类来源与召回、邻域空率/数量、刚体拟合残差、注入范数、梯度、时间、SR/LSCR和固定badcase视频。效果差不多且开销增加明显的组不优先合并；架构更复杂本身不是收益。

v2：本轮开发止于E-rel（R4/R5）通过上述验收并提交从头训40k；4卡50步冒烟后上8卡，step时间增加需<5%。D1视robotframe结果；A/C视vw25与R3/R4结果，B本轮不做。判定标准见实验方案§7（30–40k合并600局，R4相对M3 ≥ +4 pp且p<0.05）。

## 10. 当前代码入口（修改前定位）

- [dataset读取](../cosmos_framework/data/generator/action/datasets/sim_pointfk_dataset.py:19)、[PointFlowBatch字段和设备迁移](../cosmos_framework/data/pointflow_batch.py:14)、[batch构造](../cosmos_framework/data/pointflow_batch.py:137)。
- [现有geometry聚合映射](../cosmos_framework/model/generator/pointflow_geometry.py:17)、[PTV3与codec当前接入](../cosmos_framework/model/generator/pointflow_branch.py:149)。
- [网络模态编码与decode顺序](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1039)、[action输入编码](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:834)、[条件action的dummy输出](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:882)。
- [packing span定义](../cosmos_framework/data/generator/sequence_packing/modality.py:20)、[PF payload](../cosmos_framework/model/generator/pointflow_sequence.py:13)。
- [现有URDF计算](../cosmos_framework/model/generator/robot_fk_torch.py:56)、[当前刚体拟合](../cosmos_framework/model/generator/robot_fk_torch.py:165)。
- [UV resize/pad组合](../cosmos_framework/data/generator/action/pointflow_source.py:239)、[不能用于前向的GT依赖loss helper](../cosmos_framework/model/generator/pointfk_consistency.py:108)。

以上行号对应本轮核查版本，改代码后可能移动；函数名和职责是实际定位依据。
