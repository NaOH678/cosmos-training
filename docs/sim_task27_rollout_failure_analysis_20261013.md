# task27仿真rollout失败分析（2026-10-13）

## 结论

- 数据：task27 iter 20000，raw(WAM)/nolocal/local各200个rollout（4扰动×50），SR 18.0/25.0/16.5。nolocal/local保存了每个chunk的预测FK与PointFlow，raw只存video latent，故raw只能做动作–实测（tracking）分析。
- 对齐：action[t]驱动qpos[t]→qpos[t+1]，必须用qpos[r+t+1]比较。按此对齐tracking差（AR）为**7.6 mm**；此前引用的“~16 mm稳态偏差”是错开1步（qpos[r+t]）的结果（本数据该口径15.1–15.3 mm）。
- 三个差距（nolocal/local，mm）：action–dream(AD)约21、dream–reality(DR)约19.7、AR 7.6。AD/DR随episode进行变大（nolocal，chunk 0–9→20+：AD 18.9→23.0，DR 16.7→22.1），失败episode比成功episode大约3–5 mm；**检测到的失败事件前1–3个chunk没有任何差距变大**（反而低2–3 mm，但该事件检测器不可靠，见§5）。
- 球位姿pose原点不是球心（soccer 7.5 cm、tennis 5.6 cm、golf 3.8 cm、ping-pong 4.7 cm偏移）。早先基于原点高度的“drop/lift”事件含大量假阳性：16个抽样drop目视仅5个是“抓住后掉落”。
- 校正球心后的逐球漏斗：soccer主要损失在“举起→送到篮子上方”（每模型57/57/73局）；tennis在接近与举起；golf主要在“触碰→举起”（90–108局）；ping-pong几乎全被抓起并送到篮子，但只有约一半被记为placed。
- ping-pong/golf“送到篮子仍未placed”（67–80/31–33局）是**阶段依赖闸门**：`place_golf`、`place_pingpong`都`depends_on [place_soccer, place_tennis]`。325个案例中323个因soccer和/或tennis未placed而不计分，球本身物理上静止在篮内；只有2个（raw golf）是球后来离开了篮子。篮子几乎每局都被移动约40 cm，篮子位移<30 cm的局成功率为0。

## 1. 数据与产出

| 模型 | run目录（4090，`/oss-shichaojian/outputs/infer/task27/`） | SR | LSCR |
| --- | --- | --- | --- |
| raw (WAM) | `task27_cosmos_raw_20000/runs/eval_200_8gpu_shared` | 18.0% | 49.6% |
| nolocal | `task27_cosmos_nolocal_20000/runs/eval_200_8gpu` | 25.0% | 51.6% |
| local | `task27_cosmos_local_20000/runs/eval_200_8gpu` | 16.5% | 46.4% |

- 每个run：`<none|cov_only|inv_only|inv_cov>/episode_NNNNNN/{result.json, attempt_*}`。部分episode的attempt_1崩溃（exit=1，无数据），取最后一个含hdf5的attempt。
- `result.json`：success、steps（60 Hz物理步，hdf5行数=steps/3）、`stage_first_completion_step`（四个place阶段）、`violation_counts`（drop_to_table/high_speed，无时间戳）。
- `attempt_*/episodes/{success,failure}/*.hdf5`（20 Hz）：`action/commanded`(N×52绝对关节目标)、`robot/qpos`(N×52)、`objects/*/pose_world`(N×7，xyz+四元数xyzw)、`lin_vel_world`、相机rgb。
- `attempt_*/predictions/*/query_XXXXXX.npz`（仅nolocal/local）：`fk_anchor_xyz`(42×3)、`fk_displacement`(32×42×3)、`pointflow_anchor_xyz`(1024×3)、`pointflow_displacement`(32×1024×3)、`executed_actions`(32×52)；坐标`fixed_optical_camera`，单位m，位移是相对当前anchor的累计量。raw只有`query_*.latent.pt`。
- 每32行一次query，第k个chunk起始行r=32k；`executed_actions`与`commanded[r:r+32]`逐位相同。
- 未找到task27训练/示教hdf5（config指向的`unused_training_paths`不存在），示教对照未做。

## 2. 方法

- FK：`utils/sim_robot_profile.profile()/positions()`（multi_ur5_wuji_with_flange，每手21个link原点，共42点，左手0–20、右手21–41，tip为4,8,12,16,20），机器人基座系，单位m。
- 相机：对第0个query的左手21个anchor做刚体拟合得到optical←robot；所有query、左右手anchor残差最大1.1e-7 m，据此把dream换到机器人系，三个差距在同一坐标系下算欧氏距离（mm）。
- 差距定义（chunk内t=0..31、42点取平均）：AD=FK(commanded[t])对dream[t]；DR=dream[t]对FK(qpos[r+t+1])；AR=FK(commanded[t])对FK(qpos[r+t+1])。只统计满32步的chunk（6954/6762/6999个）。
- 球心 = `pose[:3] + R(quat_xyzw)·object_center_offset`（偏移取自`success/custom/task_27_ball_box_loading.py`，与USD包围盒中心一致）。半径取USD包围盒：soccer 76.8、tennis 33.5、golf 21.3、ping-pong 46 mm；篮子包围盒357×150×263 mm（局部x×y×z，y向上），中心偏移[0,0.075,0]。

## 3. 三个差距（lag-1对齐）

| 模型 | AD 均值/中位/p90 | DR 均值/中位/p90 | AR 均值/中位/p90 | AR(lag0) | 实际位移/chunk |
| --- | --- | --- | --- | --- | --- |
| raw | – | – | 7.6 / 6.5 / 11.7 | 15.1 | 71.8 |
| nolocal | 21.2 / 19.9 / 29.7 | 19.6 / 18.3 / 28.7 | 7.6 / 6.6 / 11.7 | 15.3 | 72.6 |
| local | 21.4 / 20.0 / 30.2 | 19.8 / 18.3 / 29.4 | 7.6 / 6.5 / 11.4 | 15.2 | 71.1 |

- chunk末步（t=31）：AD 23.4/23.3，DR 22.2/22.3（nolocal/local）；左右手差别≤1 mm。
- 成功/失败episode（AD/DR/AR均值）：nolocal 18.4/15.8/8.5 vs 21.8/20.4/7.5；local 18.5/16.2/8.2 vs 21.8/20.3/7.5；raw的AR 8.4 vs 7.4。AD/DR随chunk序号上升，故与episode长度混杂。
- 与实际运动量的相关：AD −0.04、DR −0.15、AR +0.2。
- 阶段完成前2个chunk AR升到11–12 mm（手臂最快时），AD/DR不升。

## 4. 对失败事件的时间锁定（含不可靠部分）

按“drop检测器”（球原点高度）对chunk相对drop块的偏移求均值（nolocal，AD/DR/AR，mm）：偏移−3：22.1/20.2/8.4；−2：20.3/19.6/7.7；−1：19.9/17.9/8.1；0：20.0/18.1/7.3；+1：19.8/18.9/6.7；无drop：21.3/19.7/7.6。同episode内配对（−2..0对其余）：AD −2.0、DR −2.1、AR +0.2（local：−2.9/−3.0/0.0）。

- 球–dream对比（球心位移，t=31，nolocal/local）：中位3D误差2.0/2.4 mm（多数球静止），均值23.3/24.3 mm，dz相关0.83/0.80。该部分用pose原点而非球心，仅供参考。
- drop事件的周期位置、边界跳变、握持闭合量、恢复率等细项也基于旧检测器，不作为结论使用。

## 5. drop检测器的坑与目视验证

- 旧检测器：球原点高于初始3 cm→回到1.2 cm内且离篮子中心>12 cm。问题：①原点≠球心，球滚动时原点高度可变化数cm（golf偏移3.8 cm）；②篮子内球在篮子局部坐标内可远于12 cm；③篮壁/篮沿上的小球也被计为“举起”。
- 目视验证（16个drop分层抽样 + 4个成功，帧：lift−10/lift/lift+10/落地起点/+10，ball居中裁剪）：

| 类别 | 数量 | 说明 |
| --- | --- | --- |
| (a) 抓住后掉落 | 5 | soccer双手握持(3)、ping-pong单手(2) |
| (b) 抓空/被手拨动 | 2 | golf |
| (c) 球在篮壁/篮沿上或靠壁 | 5 | golf |
| (d) 检测误报（球已在篮内） | 2 | soccer、ping-pong |
| (e) 其它/不确定 | 2 | golf，球小且被手遮挡 |

- 4个成功样本全是真实抓取（soccer双手、tennis单手）。golf占16例中的9例，无一属于“抓住后滑落”。
- soccer为双手抓取，按“最近单手”分配会使指尖计数失真。

## 6. 逐球漏斗（校正球心）

定义：approached=手中心（该手21点均值）到球表面≤6 cm；touched=球心离初始位置>2 cm且此时有关键点距球面≤3 cm（近4行内）；lifted2=球心高于静止3 cm且同手≥2个指尖距球面≤30 mm（soccer可用双手中心≤8 cm），连续≥5行且起点在篮子占地外；carried2=其间球心进入篮子占地（357×263 mm）；placed=`result.json`阶段完成。严格版（指尖≤15 mm）目视6/6真阳性，但漏掉真实握持（tennis/ping-pong各3/3），故以30 mm版为主；30 mm版4/4随机样本真阳性。

| 模型 | 球 | 接近 | 触碰 | 举起 | 送到篮上 | placed |
| --- | --- | --- | --- | --- | --- | --- |
| raw | soccer | 200 | 200 | 184 | 127 | 132 |
| raw | tennis | 181 | 160 | 120 | 116 | 124 |
| raw | golf | 172 | 149 | 71 | 68 | 36 |
| raw | ping-pong | 199 | 199 | 195 | 172 | 105 |
| nolocal | soccer | 200 | 200 | 186 | 129 | 133 |
| nolocal | tennis | 175 | 160 | 128 | 119 | 127 |
| nolocal | golf | 184 | 170 | 94 | 86 | 55 |
| nolocal | ping-pong | 200 | 200 | 198 | 169 | 98 |
| local | soccer | 200 | 200 | 185 | 112 | 121 |
| local | tennis | 173 | 158 | 120 | 116 | 121 |
| local | golf | 186 | 161 | 78 | 69 | 36 |
| local | ping-pong | 200 | 200 | 196 | 173 | 93 |

各环节损失局数（raw/nolocal/local）：
- soccer：举起→送到篮上57/57/73（31–40%的已举起soccer到不了篮子）。
- tennis：未接近19/25/27，触碰→举起61/47/53。
- golf：未接近28/16/14，触碰→举起101/90/108，送到→placed 32/31/33。
- ping-pong：触碰→举起仅2–4，举起→送到23/29/23，送到→placed 67/71/80。
- placed多于送到：soccer 8–11、tennis 7–9局是球进篮时未满足lifted2/起点条件（如被推入）。

## 7. 篮子位移

| 模型 | 位移最大值 xy中位/p90/p99 (mm) | 终点位移中位 | 最大yaw中位/p90 (°) | 位移>2 cm局数 |
| --- | --- | --- | --- | --- |
| raw | 429 / 482 / 651 | 402 | 21.1 / 37.7 | 180/200 |
| nolocal | 415 / 463 / 504 | 394 | 19.5 / 30.0 | 181/200 |
| local | 421 / 470 / 670 | 396 | 20.3 / 29.9 | 175/200 |

- 成功率按篮子最大位移分档（raw/nolocal/local，局数）：0–20 mm 0%(20/19/25)；20–300 mm 0%(15/19/23)；300–400 mm 19/35/25%(16/34/20)；400–450 mm 33/34/27%(89/89/83)；>450 mm 7/21/12%(60/39/49)。位移与placed球数的相关0.57/0.57/0.62。
- 即策略把篮子拖向球约40 cm，不移动篮子的局全部失败。
- golf/ping-pong未placed时：球到篮子占地的初始距离中位153（golf）/78–80 mm（ping-pong），首次接近157/85–87，首次触碰154–157/106–107；触碰时距篮壁<30 mm的比例≤1%，接近前被推动>2 cm的比例≤2%。球不是被推靠到篮壁上。

## 8. “送到篮子但未placed”：阶段依赖闸门

`scenes/27_ball_box_loading.yaml`的stages（实现`benchmark/stage_tracker.py`、`success/condition_evaluator.py`）：
- 阶段谓词 = 篮子局部y轴与世界z夹角<30° ∧ 球心（pose+R·offset）到容器中心（篮子pose+R·[0,0.075,0]）水平距离<0.22 m且球心z>容器中心z−0.05 m ∧ 球线速度<0.05 m/s（角速度不查）。无手接触/释放条件，无最大高度，无停留时间。
- 锁存：谓词首次为真且依赖阶段此前已锁存（或同一步变真）时记完成。`place_soccer`、`place_tennis`无依赖；`place_golf`、`place_pingpong`依赖二者。整局成功还需四球同时在篮内且静止。

对325个“送到篮子但未placed”的golf/ping-pong案例，沿时间评估各条件：

| 模型 | 球 | n | 仅缺soccer | 仅缺tennis | 两者都缺 | 都已placed（球后来离开篮） |
| --- | --- | --- | --- | --- | --- | --- |
| raw | golf | 32 | 4 | 7 | 19 | 2 |
| nolocal | golf | 33 | 10 | 13 | 10 | 0 |
| local | golf | 33 | 7 | 8 | 18 | 0 |
| raw | ping-pong | 72 | 9 | 14 | 49 | 0 |
| nolocal | ping-pong | 73 | 21 | 27 | 25 | 0 |
| local | ping-pong | 82 | 21 | 19 | 42 | 0 |

- 全部案例中球自身谓词（篮内、篮子直立、静止）在某些行为真，但依赖阶段当时都未完成；自身谓词为真而依赖缺失的行数中位约800（golf）/960（ping-pong），约等于该局剩余时间。
- 终点状态：球仍在0.22 m区域内golf 29–32、ping-pong 70–80，终点静止（中位速度≤0.003 m/s），均无手在3 cm内（第2近指尖对球面），球在篮沿之上1例、叠在另一球上1–2例。
- 2个例外（raw golf，none/21、none/47）：soccer、tennis已placed，但golf随后离开篮子（球心比容器中心低5.6/8.2 cm，z界失败；none/21球在篮外桌面上）。
- 目视4例（首次进入/+60/中段/终段，篮子居中裁剪）：raw ping-pong两者都缺、nolocal golf缺tennis、local ping-pong缺soccer、raw golf在篮外，均与上表一致。

### 每球“物理在篮内”与“计分”对比（局数/200，终点状态）

| 模型 | 球 | 终点在篮内 | 篮内+直立+静止 | 计分placed |
| --- | --- | --- | --- | --- |
| raw | soccer / tennis | 128 / 124 | 126 / 123 | 132 / 124 |
| raw | golf / ping-pong | 65 / 176 | 65 / 171 | 36 / 105 |
| nolocal | soccer / tennis | 126 / 126 | 121 / 125 | 133 / 127 |
| nolocal | golf / ping-pong | 88 / 176 | 88 / 173 | 55 / 98 |
| local | soccer / tennis | 109 / 116 | 103 / 114 | 121 / 121 |
| local | golf / ping-pong | 68 / 177 | 67 / 173 | 36 / 93 |

- ping-pong终点物理在篮内约173/200，计分只有93–105；golf 65–88 vs 36–55；soccer/tennis计分略高于终点在篮内（阶段已锁存，球之后被挪出）。
- 因此golf/ping-pong的“送到→placed”损失主要由soccer（举起→送到篮上57–73局）和tennis（接近+举起约66–80局）两级的失败传递而来。

## 9. 注意事项

- 20 Hz行评估闸门条件，评估器在60 Hz；自身谓词为真的行数多达数百，不影响计数结论。
- 手的距离用FK link原点，不是表面；球半径为包围盒估计，篮子占地用外包围盒（含壁厚）。
- 手分配取“lift时最近的手”，soccer为双手抓取，单手指尖计数不可靠。
- 球–dream对比、drop周期位置、边界跳变、闭合量、carry运动学的AUC（palm–ball距离0.84等）均基于原点高度的旧事件，不应引用为失败机理；§3、§6–§8不受影响（只用机器人FK或校正球心）。
- raw/WAM无dream几何，未解码latent。

## 10. 脚本

4090：`/root/claude_tmp/gap_20261013/`（仅CPU分析，读取`/oss-shichaojian/outputs/infer/task27/`，输出在该目录）。

| 脚本 | 作用 |
| --- | --- |
| `extract.py` / `extract2.py` | 每episode抽取FK、相机拟合、AD/DR/AR与球位移 → `out/*.npz`（extract2含n<0修正） |
| `analyze.py`、`extra.py` | §3、§4汇总表 → `tables.pkl` |
| `follow.py`、`follow_an.py` | drop周期位置、边界跳变、闭合量、恢复率（旧检测器，仅参考） |
| `carry.py`、`carry_an.py` | drop/成功carry的运动学、抓取构型、AUC（旧检测器，仅参考） |
| `frames.py`…`frames6.py`、`mksel.py` | 目视验证帧拼图 → `frames*/` |
| `funnel3.py`、`funnel_an3.py`、`funnel_an2.py` | 校正球心的逐球漏斗、篮子位移、进篮方式 → `funnel3.pkl` |
| `gate.py`、`gate_an.py`、`anom.py`、`tilt.py`、`enter.py`、`phys.py` | 阶段依赖闸门、例外案例、篮子倾角、终点物理在篮内统计 |
| `geo.py`、`quat.py` | USD包围盒（球半径/篮子占地）、四元数约定检查 |
