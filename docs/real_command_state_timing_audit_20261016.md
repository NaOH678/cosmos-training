# 真机指令与实际状态时间对应核查（2026-10-16）

## 结论

数据包含“同时刻附近的指令和状态”，并非“这一行指令执行完后的状态”。
原始数据同步大体良好，但存在物理跟踪延迟；不能照搬仿真的一控制步对应关系。
另发现早期EEF版action的手腕字段实际等于观测EEF，不是诊断字段中的目标EEF。
本次只读检查原始数据与缓存，未修改训练数据、模型或对齐逻辑。

## 范围与依据

- 时间戳：sandwich 101条/106731行、dropper 101条/107703行、dropper mix 71条/72788行，共273条。
- 每种数据均匀抽15条，共45条，读取原始LMDB核查字段；以原始通道时间戳估计运动延迟。
- 原始根目录：`/data/shichaojian/raw_data/{singlerighthand_sandwich_100,singlerighthand_dropper_100,dropper_dagger_mix}/`。
- 原始字段：`/observations/qpos`、`/diagnostics/arm_joint_command`、`action`、`/observations/eef`、`/diagnostics/commanded_eef`、`/observations/hand_joint_deg`。
- 每条episode的`sync_timestamps.json`按camera_head为anchor保存各来源时间；元数据来源为wuji-hand-teleop/ros2。
- 代码与结果：`pointflow_outputs/real_timing_audit_20261016/{audit.py,raw_check.py,report.json,raw_report.json}`。
  延迟扫描用已知70ms合成信号验证正负号，通过。

## 1. 同一行是否同一时刻

三组数据anchor间隔中位数均33.36ms，metadata均30fps。
同一行arm/hand指令时间减实际状态时间，中位数在-0.28至-0.83ms；
各组10%—90%范围约-4.5至+3.4ms。说明大多数记录确为接近同一时刻，
而不是状态字段预先整体错开一两行。

但同步不是严格硬件采样同步：存在异常间隔，三组最大相邻anchor间隔分别为
100ms、300ms、1067ms；这里未核对设备端时钟/传输耗时，不能将记录时间当作精确电机生效时间。

## 2. 真实运动落后指令多少

方法：对有明显变化的右侧关节，按各自来源时间戳插值，在固定公共时间范围内
扫描-200至+600ms，10ms间隔，比较指令与状态的一阶变化相关性；
正延迟表示command(t)与state(t+延迟)最相近。
以下为每组15条的episode中位数，仅为响应延迟的统计估计，原始30Hz采样不支持10ms精度断言。

| 数据 | 手臂关节指令→实际关节 | 手指指令→实际关节 | 目标EEF位置→观测EEF位置 |
| --- | --- | --- | --- |
| sandwich | 约60ms（样本范围60—80） | 约40ms（30—50） | 约80ms（80—90） |
| dropper | 约70ms（70—90） | 约30ms（30—40） | 约90ms（80—100） |
| dropper mix | 约70ms（60—70） | 约30ms（30—50） | 约90ms（80—120） |

手臂关节变化相关系数中位数约0.96—0.98，手指约0.74—0.84。
手臂/手指响应不同，不能统一把整个27维action或FK标签平移相同帧数。
这些是记录信号间的有效延迟，不足以拆分控制器动态、滤波与传输各自占比。

## 3. 当前训练怎样对应

`tools/prepare_singlerighthand_raw.py`逐行读LMDB并写state/action，没有用sync时间戳重新插值。
`SinglerighthandRawDataset._window_indices`返回：

```text
观测索引：s, s+d, s+2d, ..., s+32d
动作索引：s, s+d, ..., s+31d
模型action第0行：state[s]
模型action第k行：action[s+(k-1)d]，k=1..32
FK第k个未来状态：实际位置[s+kd]
```

因此，以30Hz源数据、15Hz训练为例，d=2，action与对应未来FK相隔约66.7ms。
这接近手臂约60—70ms的有效延迟，但与手指约30—40ms不同；并非证明所有关节严格对齐。
窗口内部只取每个观测区间起点的一份command，不能代表区间内所有连续发出的控制指令。
源文件偶发丢帧/长间隔也不会在该reader里按真实时间修正。

## 4. 重要字段语义：EEF版action不是全部都是指令

抽查每组15条，45/45条的`action[:,27:34]`与`/observations/eef[:,7:14]`逐元素完全相同。
它们与`/diagnostics/commanded_eef[:,7:14]`并不相同；后者与观测EEF有约80—90ms的变化延迟。

这意味着：

- 早期sandwich EEF缓存读取原始action的手腕字段，手腕监督实质来自实际观测轨迹；
  手指部分仍读取目标角度。此前将其整体称为“目标手腕＋手指目标”不准确。
- 当前普通dropper/mix的joint缓存由`/diagnostics/arm_joint_command`取手臂目标，
  由原始action取手指目标；state由qpos获取。这个路径确实区分关节指令和实际状态。
- 原始sandwich也有arm_joint_command和qpos，可以另建joint缓存；不能因此把现有EEF缓存当joint模型使用。

这里只证明已存字段的相等关系；未取得采集端ROS源码，不推断采集程序为何选择这种action定义。

## 5. 对拟议action＋FK残差方案的影响

1. joint版真机数据可以构造command-FK与实际FK，但必须明确比较的时间。
2. 若预测目标是未来时刻的实际FK，应保留真实执行延迟作为残差的一部分；
   不应为了让误差变小而先按最大相关把所有滞后消掉。
3. 若要单独研究“不含延迟的跟踪误差”，才另做按时间插值的对齐诊断，并分别处理手臂与手指。
4. 模型可利用当前实际状态及历史/预测指令解释执行残差；残差不是单个未来指令的固定修正量。
5. 目前证据不足以要求重训或整体平移标签；先确定目标时间契约，再讨论修改数据。
