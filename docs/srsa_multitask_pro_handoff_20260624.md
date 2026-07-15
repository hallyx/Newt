# SRSA 多任务训练外部评审交接说明

日期：2026-06-24

维护约定：本文档后续更新统一使用中文；代码标识、路径、配置项和日志名保持原文。

本文整理当前 SRSA/Newt 多任务训练状态，供外部评审判断：在加入更多 assembly 之前，应该如何修改多任务条件注入路径。

## 摘要

当前主线是 staged online family replay V2（acquisition-first）：

- 每个 online train job 仍然只使用一个 active assembly env；
- 旧 assembly 通过 replay 进入训练，而不是通过真正的 mixed-assembly IsaacLab env；
- `01125` 是 retention anchor；
- `00256` 是最干净的第二任务诊断目标；
- 在 `01125 + 00256` 同时通过 retention 和 task-vector sensitivity 之前，不加入 `00186`。

当前核心结论：

```text
00256 可学。
01125 + 00256 可以保持较高 relaxed success。
但 baseline 基本忽略 task_vec_6。
raw_task_vec 可以让模型对 task_vec_6 产生敏感性。
full-site adapter 太破坏策略/价值/规划表面。
只作用在 encoder+dynamics 的弱 raw_task_vec adapter 是目前第一个不破坏性能的正向结果。
alpha=0.01 会进一步增强离线 sensitivity，但闭环 realistic swap 仍没有明显拉开。
```

下一步设计问题不是“是否加入第三任务”，而是：

```text
如何注入 task_vec_6，让它影响任务路由，同时不破坏 policy/value/planning surface？
```

## 当前代码路径

### adapter 之前的任务条件路径

正常路径：

```text
task_vec_6 -> AxialTaskEncoder -> task_context_64
```

`WorldModel.task_emb()` 会把 `task_context_64` 拼进模型输入：

- encoder input：`obs + task_context_64`
- dynamics input：`z + task_context_64 + contact_context + action`
- reward input：`z + task_context_64 + action`
- policy prior input：`z + task_context_64`
- Q input：`z + task_context_64 + action`

相关文件：

- `tdmpc2/common/world_model.py`
- `tdmpc2/models/axial_task_encoder.py`

问题：离线 paired sensitivity 显示，当前学到的 `AxialTaskEncoder` context 对真实 SRSA 向量几乎塌缩。只改变 `task_vec_6` 时，action/Q/reward/next-latent 差异接近数值噪声。

### Task context FiLM adapter

已加入 zero-init FiLM adapter，由以下配置控制：

```text
task_context_adapter_enabled=true
```

公式：

```text
y = x + alpha * (gamma(task) * x + beta(task))
```

adapter source 可选：

- `task_context`：学到的 64D `AxialTaskEncoder` 输出；
- `raw_task_vec`：原始 6D `task_vec_6`；
- `both`：二者拼接。

配置和 online-family launcher 已暴露 site flags：

```text
task_context_adapter_apply_encoder
task_context_adapter_apply_dynamics
task_context_adapter_apply_policy
task_context_adapter_apply_reward
task_context_adapter_apply_q
```

当前最有信息量的诊断设置：

```text
source = raw_task_vec
alpha = 0.005 or 0.01
encoder = true
dynamics = true
policy = false
reward = false
q = false
lr_scale = 0.1
```

### task conditioning 如何影响动作

SRSA policy 使用 MPC/MPPI。task vector 可以通过以下路径影响最终动作：

```text
obs + task -> encode -> z0
z0 + task -> pi -> policy action trajectories
z + action + task -> dynamics -> predicted next latent
z + action + task -> reward -> trajectory reward
z + action + task -> Q -> terminal value
MPPI ranks candidate action sequences using reward + Q
```

因此，如果直接调制 Q/reward/dynamics，会强烈改变 planner objective。这也是 full-site raw-task adapter 在 `alpha=0.05` 下敏感性增强但成功率崩掉的主要嫌疑。

## 实验记录

### 1. Direct fine-tune baseline

这些实验用于确认任务难度，并证明 bridge/training loop 能学会部分 target assembly。

| Target | Best success | Step | 解释 |
| --- | ---: | ---: | --- |
| `00256` | `0.9556` | `600k` | 明确可学 |
| `00186` | `0.4133` | `600k` | 部分可学，但更难 |
| `00062` | `0.0` | `150k` | 困难样本 |
| `00271` | `0.0` | `150k` | 困难样本 |

补充 `00256` 检查：

| Run | Best success | Step |
| --- | ---: | ---: |
| single from old `01125` checkpoint | `0.9688` | `399,360` |
| single from online-family stage-1 checkpoint | `0.8867` | `299,520` |

结论：`00256` 失败不是因为任务不可学。

### 2. 早期 online-family replay 尝试

早期 online-family run 过早 handoff。只在 `00256` 上训练约 50k steps 后，family eval：

| Eval after `00256` | relaxed success |
| --- | ---: |
| `01125` | `0.90` |
| `00256` | `0.00` |

随后加入 `00186` 也失败：

| Eval after `00186` | relaxed success |
| --- | ---: |
| `01125` | `0.65` |
| `00256` | `0.00` |
| `00186` | `0.00` |

结论：这次 50k-stage 失败不应视为多任务训练不可行，主要说明 handoff 太早。

### 3. V2 acquisition-first：`01125 -> 00256`

Run:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_launcher
```

最佳 `00256`：

```text
episode_success = 0.9023 at 299,520 steps
```

`00256` 后 retention：

| Assembly | relaxed success | strict success | process success |
| --- | ---: | ---: | ---: |
| `01125` | `0.75` | `0.20` | `0.20` |
| `00256` | `0.90` | `0.00` | `0.00` |

同 checkpoint 的 task-vector swap：

| 00256 env input task vector | relaxed success |
| --- | ---: |
| correct `00256` vector | `0.90` |
| forced `01125` vector | `0.90` |
| forced zero vector | `0.75` |

结论：成功率本身不能证明模型使用了 task conditioning。

### 4. Replay/storage/sampling 检查

Replay snapshots：

```text
20260615_202326_launcher/replay/01125.pt
20260618_001734_launcher/replay/00256.pt
```

二者都存储 per-transition task tensor，shape 为 `[N, 6]`。

一次 50/50 online-family mixed sample 返回：

```text
task.shape == [3, 64, 6]
task counts == {'01125': 32, '00256': 32}
```

结论：没有证据表明 replay 层存在全局 task-vector broadcast bug；task tensor 能到达 update path。

### 5. Adapter 前的离线 paired sensitivity

Report:

```text
logs/task_vec_sensitivity/20260619_00256_v2_offline_report.json
```

只改变 `task_vec_6` 的 mean deltas：

| Swap | action L2 | Q abs | reward abs | next-latent L2 |
| --- | ---: | ---: | ---: | ---: |
| `01125` vector | `4.43e-7` | `2.09e-5` | `9.81e-7` | `1.84e-6` |
| old `00186` vector | `4.76e-7` | `2.18e-5` | `9.42e-7` | `1.92e-6` |
| zero vector | `4.79e-7` | `2.11e-5` | `9.78e-7` | `1.91e-6` |

结论：模型对真实 task-vector 改变几乎不变。

### 6. 50/50 retention polish

Run:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_polish_01125_00256/20260619_0135_retention_polish_launcher
```

训练：

```text
best 00256 episode_success = 0.9609 at 99,840 steps
```

Retention：

| Assembly | relaxed success | strict success | process success |
| --- | ---: | ---: | ---: |
| `01125` | `0.90` | `0.20` | `0.25` |
| `00256` | `0.95` | `0.00` | `0.00` |

Mixed updates 精确 50/50：

```text
01125 = 512
00256 = 512
entropy_norm = 1.0
```

paired sensitivity 仍接近噪声：

| Swap | action L2 | Q abs | reward abs | next-latent L2 |
| --- | ---: | ---: | ---: | ---: |
| `01125` vector | `4.54e-7` | `3.22e-5` | `2.06e-6` | `2.08e-6` |
| zero vector | `4.61e-7` | `3.18e-5` | `2.22e-6` | `2.10e-6` |

结论：replay balance 和 retention 可以工作，但不会自动让 `task_vec_6` 变成不可忽略信息。

### 7. Learned task-context adapter, alpha 1.0

Run:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_adapter_01125_00256/20260622_taskctx_adapter_00256_launcher
```

设置：

```text
task_context_adapter_enabled=true
task_context_adapter_source=task_context
task_context_adapter_alpha=1.0
```

训练结果：

```text
00256 best episode_success = 0.5781 at 99,840 steps
```

Family eval：

| Assembly | relaxed success | official latched | mean lateral error |
| --- | ---: | ---: | ---: |
| `01125` | `0.00` | `1.00` | `83.1 mm` |
| `00256` | `0.00` | `0.40` | `9.47 mm` |

Sensitivity 仍接近噪声：

| Swap | action L2 |
| --- | ---: |
| `01125` vector | `1.73e-7` |
| old `00186` vector | `1.82e-7` |
| zero vector | `1.71e-7` |

后续诊断发现 learned task context 本身几乎塌缩：

```text
00256 vs 01125 task-context L2 ~ 1.8e-8
00256 vs 00186 task-context L2 ~ 4.2e-8
```

结论：如果 learned context 不区分任务，基于它调制 adapter 没有意义。

### 8. Raw task-vector adapter, alpha 0.05

Run:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/20260622_194546_launcher
```

设置：

```text
task_context_adapter_enabled=true
task_context_adapter_source=raw_task_vec
task_context_adapter_alpha=0.05
full-site adapter: encoder, dynamics, policy, reward, Q
```

训练结果：

| Step | 00256 episode_success |
| ---: | ---: |
| pre-update rollout | `0.9492` |
| `49,920` eval | `0.1445` |
| `99,840` eval | `0.0000` |

Family eval：

| Assembly | relaxed success | strict success | official latched | mean lateral error |
| --- | ---: | ---: | ---: | ---: |
| `01125` | `0.00` | `0.00` | `0.85` | `103.1 mm` |
| `00256` | `0.00` | `0.00` | `0.25` | `98.4 mm` |

采样正确：

```text
01125 = 512
00256 = 512
entropy_norm = 1.0
```

Adapter norms 快速增长：

| Site | final weight norm |
| --- | ---: |
| encoder | `7.92` |
| dynamics | `8.15` |
| policy | `7.63` |
| reward | `5.91` |
| Q | `54.21` |

修正后的 sensitivity report：

```text
logs/task_vec_sensitivity/20260622_rawtask_adapter_00256_report.json
```

该 report 是在修复 `tdmpc2/scripts/task_vec_sensitivity_report.py` 之后重跑的；修复点是从 checkpoint metadata 恢复 `task_context_adapter_alpha`。

Mean deltas：

| Swap | action L2 | Q abs | reward abs | next-latent L2 |
| --- | ---: | ---: | ---: | ---: |
| `01125` vector | `0.0595` | `1.64` | `0.227` | `0.562` |
| old `00186` vector | `0.0752` | `22.21` | `0.393` | `1.170` |
| zero vector | `0.0888` | `5.33` | `0.437` | `1.257` |
| random vector | `0.1025` | `28.96` | `0.410` | `1.532` |

结论：

```text
raw_task_vec 修复了 invariance 症状，
但 alpha 0.05 的 full-site adapter 太破坏性。
```

### 9. Raw task-vector adapter, alpha 0.005, only encoder+dynamics

Run:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/20260624_2115_rawtask_alpha0005_sitelimited_launcher
```

设置：

```text
task_context_adapter_enabled=true
task_context_adapter_source=raw_task_vec
task_context_adapter_alpha=0.005
task_context_adapter_lr_scale=0.1
adapter sites: encoder, dynamics
disabled sites: policy, reward, Q
```

训练结果：

| Step | 00256 episode_success |
| ---: | ---: |
| `49,920` eval | `0.9219` |
| `99,840` eval | `0.9453` |

最终 adapter 输出影响较小：

| Site | weight norm | delta L2 mean | relative delta norm |
| --- | ---: | ---: | ---: |
| encoder | `0.927` | `0.0258` | `0.0059` |
| dynamics | `1.107` | `0.0311` | `0.0072` |

Family eval：

| Assembly | relaxed success | strict success | process success | mean lateral error |
| --- | ---: | ---: | ---: | ---: |
| `01125` | `0.90` | `0.15` | `0.15` | `0.904 mm` |
| `00256` | `0.95` | `0.15` | `0.15` | `0.627 mm` |
| family mean | `0.925` | `0.15` | `0.15` | `0.765 mm` |

采样保持精确 50/50：

```text
01125 = 512
00256 = 512
condition entropy norm = 1.0
task hashes = 28e69f22e900:512, daa27bb0a1ac:512
```

Sensitivity report：

```text
logs/task_vec_sensitivity/20260624_rawtask_alpha0005_sitelimited_00256_report.json
```

Mean deltas：

| Swap | action L2 | Q abs | reward abs | next-latent L2 |
| --- | ---: | ---: | ---: | ---: |
| `01125` vector | `3.70e-4` | `4.14e-3` | `1.14e-3` | `8.11e-4` |
| zero vector | `8.68e-3` | `1.10e-1` | `2.81e-2` | `2.26e-2` |
| random vector | `1.65e-2` | `2.09e-1` | `5.34e-2` | `4.54e-2` |
| extreme vector | `6.38e-4` | `7.97e-3` | `2.00e-3` | `1.39e-3` |

结论：

```text
site-limited raw adapter 保住了 01125/00256 性能，
并产生了高于数值噪声的 sensitivity。
但真实 01125-vs-00256 sensitivity 仍偏弱，
因此这是正向诊断结果，不是 task_vec_6 已不可替代的证明。
```

### 10. Successful site-limited adapter 的闭环 task-vector swap

Run:

```text
logs/task_vec_swap_eval/20260624_2220_sitelimited_swap_eval
```

Checkpoint：

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/20260624_2115_rawtask_alpha0005_sitelimited_stage-2_asm-00256/models/best_step-99840_s-0p9453.pt
```

该 eval 前修复了兼容路径：

```text
tdmpc2/collect_eval_rollouts.py 会在 batch eval compatibility matching 时，
从 checkpoint metadata 恢复 task_context_adapter_alpha、
task_context_adapter_lr_scale 和 task-vector normalization metadata。
```

Eval logs 确认：

```text
adapter_source=raw_task_vec
adapter_alpha=0.005
sites=['encoder', 'dynamics']
```

`00256` env 结果：

| Model task vector | relaxed success | strict success | process success | reward | lateral error | keypoint error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| correct `00256` | `0.95` | `0.15` | `0.15` | `678.45` | `0.627 mm` | `2.186 mm` |
| forced `01125` | `0.95` | `0.10` | `0.10` | `670.61` | `0.633 mm` | `2.053 mm` |
| forced zero | `1.00` | `0.15` | `0.15` | `695.72` | `0.618 mm` | `1.902 mm` |

结论：

```text
alpha 0.005 encoder+dynamics adapter 不破坏性能，
离线 sensitivity 也高于数值噪声，
但 00256 闭环成功率仍不依赖正确 task vector。
forced 01125 不降低 relaxed success，forced zero 甚至更好。
```

### 11. 更强 site-limited raw adapter, alpha 0.01

Run:

```text
logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/20260625_0010_rawtask_alpha001_sitelimited_launcher
```

设置：

```text
task_context_adapter_source = raw_task_vec
task_context_adapter_alpha = 0.01
task_context_adapter_lr_scale = 0.1
encoder = true
dynamics = true
policy = false
reward = false
q = false
```

训练结果：

| Step | 00256 relaxed success |
| ---: | ---: |
| `49,920` | `0.7930` |
| `99,840` | `0.9531` |

最终 adapter 输出影响：

| Site | weight norm | delta L2 mean | relative delta norm |
| --- | ---: | ---: | ---: |
| encoder | `0.942` | `0.0476` | `0.0110` |
| dynamics | `1.113` | `0.0546` | `0.0127` |

Family eval：

| Assembly | relaxed success | strict success | process success | mean lateral error |
| --- | ---: | ---: | ---: | ---: |
| `01125` | `0.95` | `0.15` | `0.15` | `0.935 mm` |
| `00256` | `1.00` | `0.05` | `0.05` | `0.592 mm` |
| family mean | `0.975` | `0.10` | `0.10` | `0.764 mm` |

Sensitivity report：

```text
logs/task_vec_sensitivity/20260625_rawtask_alpha001_sitelimited_00256_report.json
```

Mean deltas：

| Swap | action L2 | Q abs | reward abs | next-latent L2 |
| --- | ---: | ---: | ---: | ---: |
| `01125` vector | `2.31e-3` | `3.17e-2` | `5.93e-3` | `5.83e-3` |
| zero vector | `2.31e-2` | `2.11e-1` | `9.70e-2` | `6.29e-2` |
| random vector | `3.06e-2` | `2.83e-1` | `1.29e-1` | `8.18e-2` |
| extreme vector | `3.14e-3` | `4.17e-2` | `8.17e-3` | `8.31e-3` |

闭环 task-vector swap：

```text
logs/task_vec_swap_eval/20260625_alpha001_sitelimited_swap_eval
```

`00256` env 结果：

| Model task vector | relaxed success | strict success | process success | reward | lateral error | keypoint error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| correct `00256` | `1.00` | `0.05` | `0.05` | `684.83` | `0.592 mm` | `1.973 mm` |
| forced `01125` | `1.00` | `0.05` | `0.05` | `680.70` | `0.600 mm` | `2.270 mm` |
| forced zero | `0.95` | `0.00` | `0.00` | `674.24` | `0.581 mm` | `2.471 mm` |

结论：

```text
alpha 0.01 保住了 acquisition 和 retention，
并把 realistic 01125 swap 的离线 sensitivity 相比 alpha 0.005 提高约 6 倍。
但闭环成功率仍几乎不受 realistic 01125-vs-00256 swap 影响。
只有 zero vector 造成小幅 relaxed-success 下降，并清空 strict/process success。
```

## 当前判断

### 大概率成立

1. `00256` 可学，应继续作为第二诊断任务。
2. `00186` 更难，不应现在用来 debug task conditioning。
3. replay/storage/update 路径对当前诊断基本可信。
4. `AxialTaskEncoder` 输出过度塌缩，不能独立承担任务路由。
5. 原始 6D task input 能让模型对任务身份产生 sensitivity。
6. 直接调制 encoder+dynamics+policy+reward+Q 对当前 checkpoint 太激进。
7. 弱 raw-task adapter 只作用于 encoder+dynamics 时，可以保住性能并把 sensitivity 推到噪声以上。
8. alpha 从 `0.005` 提到 `0.01` 会增强离线 sensitivity，同时仍能保住 retention。
9. `0.005` 和 `0.01` 都没有让 00256 闭环 relaxed success 强依赖 realistic 01125-vs-00256 task-vector 差异。

### 仍不确定

1. Q/reward/policy 是否全部有害，还是其中某个 site 可以用极小 alpha/lr 单独重新引入。
2. raw task vector 在 FiLM 或 encoder 前是否需要 family-stat normalization。
3. `AxialTaskEncoder` 应该通过辅助 loss 修复，还是应该用 raw residual path 绕过。
4. 当前 relaxed success 对 pairwise swap 是否太宽松；即使 task-conditioned predictions 有差异，relaxed success 也可能保持很高。

## 推荐下一步修改计划

### Step 0：继续固定两任务

不要加入 `00186`。

保持：

```text
01125 + 00256
CURRENT_RATIO=0.50
ANCHOR_RATIO=0.50
HISTORY_RATIO=0.0
srsa_axial_clearance_depth_templates=1.0:1.0
```

### Step 1：弱 raw-task adapter 已完成

Wrapper：

```text
scripts/run_01125_00256_rawtask_adapter_diagnostic.sh
```

默认设置：

```text
TASK_CONTEXT_ADAPTER_SOURCE=raw_task_vec
TASK_CONTEXT_ADAPTER_ALPHA=0.005
```

结果已通过 acquisition/retention gate：

```text
00256 final eval = 0.9453
01125 family relaxed success = 0.90
00256 family relaxed success = 0.95
family mean = 0.925
```

Acceptance signals：

- `00256` eval 接近 pre-adapter/polish 水平：通过。
- `01125` retention `>=0.75`：通过。
- family mean `>=0.80`：通过。
- batch task counts 50/50：通过。
- paired sensitivity 高于数值噪声：通过，但 realistic 01125-vs-00256 sensitivity 仍偏弱。

### Step 2：launcher 已暴露 adapter site flags

已完成。配置和 launcher 暴露：

```text
task_context_adapter_apply_encoder
task_context_adapter_apply_dynamics
task_context_adapter_apply_policy
task_context_adapter_apply_reward
task_context_adapter_apply_q
```

环境变量名：

```text
TASK_CONTEXT_ADAPTER_APPLY_ENCODER
TASK_CONTEXT_ADAPTER_APPLY_DYNAMICS
TASK_CONTEXT_ADAPTER_APPLY_POLICY
TASK_CONTEXT_ADAPTER_APPLY_REWARD
TASK_CONTEXT_ADAPTER_APPLY_Q
```

第一组 site-limited 设置：

```text
source = raw_task_vec
alpha = 0.005
encoder = true
dynamics = true
policy = false
reward = false
q = false
```

理由：

- encoder/dynamics 可以让 latent transition 更 task-aware；
- 直接调制 Q/reward 会重塑 MPPI objective，已表现出破坏性；
- 直接调制 policy prior 可能过早偏置 candidate trajectories。

### Step 3：adapter learning-rate control 已完成

已完成。adapter 参数独立 optimizer group：

```text
task_context_adapter_lr_scale: float = 0.1
```

理由：

- 失败的 raw-task run 中 Q adapter norm 到 `54.21`；
- 小 alpha 可能被大权重抵消；
- LR scale 提供独立于 representation strength 的控制旋钮。

### Step 4：adapter-output diagnostics 已完成

已完成。日志现在按 active site 记录：

```text
adapter_delta_l2_mean
adapter_delta_l2_p95
adapter_relative_delta_norm
```

alpha 0.005 成功 run 中 relative delta norm 约低于 `0.8%`。alpha 0.01 中 encoder/dynamics relative delta norm 约为 `1.1%`/`1.27%`。

### Step 4b：闭环 task-vector swap eval 已完成

alpha 0.005 checkpoint 的 `00256` env 结果：

```text
correct 00256 = 0.95
forced 01125 = 0.95
forced zero = 1.00
```

解释：

- 离线 paired sensitivity 高于噪声；
- realistic 01125-vs-00256 action sensitivity 只有 `3.70e-4`；
- 闭环 success 仍对 realistic task-vector swap 不敏感；
- 若目标是 task-vector indispensability，不应加入 `00186`。

### Step 4c：更强 site-limited adapter 已完成

使用：

```text
source = raw_task_vec
alpha = 0.01
lr_scale = 0.1
encoder = true
dynamics = true
policy = false
reward = false
q = false
```

结果：

```text
correct 00256 relaxed success = 1.00
01125 retention = 0.95
forced 01125 relaxed success = 1.00
forced zero relaxed success = 0.95
```

解释：

- acquisition/retention 强通过；
- offline realistic-swap action sensitivity 增至 `2.31e-3`；
- closed-loop realistic-swap success 仍没有分离；
- 若目标仍是 task-vector indispensability，不应加入 `00186`。

### Step 4d：alpha 0.01 之后的下一步

不要继续简单增大 alpha。下一步应该让 task vector 直接影响一个 supervised 或 planning-relevant quantity，同时保护 policy/value surface。

推荐下一条代码路线：

```text
raw residual path into task_context_64
task_vec reconstruction loss from task_context_64
small spread/contrastive loss across batch task contexts
optional task_vec normalization by family stats
```

这个测试中仍保持 adapter sites 只开 encoder+dynamics。不要重新开启 reward/Q/policy FiLM，除非 encoder 表征本身已经不塌缩，或者只对其中一个 site 做极小 alpha 的隔离 ablation。

### Step 5：修改 AxialTaskEncoder 本体

弱/强 site-limited raw adapter 都能保住成功率，但 realistic closed-loop swap sensitivity 仍太弱。因此下一步应回到 `AxialTaskEncoder` 本体：

- 加 raw residual path：`task_context = AxialTaskEncoder(task_vec) + Linear(raw_task_vec)`；
- 加从 `task_context` 重构 `task_vec_6` 的 reconstruction loss；
- 加 task context 的 contrastive/spread loss；
- 在 encoder/adapter 前加入 task-vector normalization。

只有在分清 adapter 强度/site selection 与 encoder collapse 的关系后，再考虑加入第三任务。

## 给外部评审的问题

1. task identity 是否应该直接影响 reward/Q，还是只影响 latent encoding 和 dynamics？
2. FiLM residual modulation 是否合适，还是 raw task vector 应该通过 additive residual/context gates 进入？
3. adapter 实验初期是否应该冻结 base world model，还是必须 joint training 才能兼容 MPC？
4. 哪种 regularizer 最能避免 task-vector shortcut 破坏已有单任务 policy？
5. `task_vec_6` 在进入 adapter/encoder 前是否应按 family statistics normalization？
6. 什么数量级的 paired sensitivity 对本任务才算有意义？baseline 是 `~1e-7`，raw full-site alpha `0.05` action L2 到 `~0.06-0.10` 但 retention 失败；site-limited alpha `0.01` realistic swap action L2 为 `2.31e-3` 但闭环仍不分离。

## 暂时不要做

- 不要把 `00186` 加入主实验。
- 不要加入 `00062` 或 `00271`。
- 不要重新开启尺寸泛化 templates。
- 不要把 official-latched success 当作充分指标；relaxed/process/strict metrics 和 lateral/keypoint errors 都要看。
- 不要只根据 success 评价 raw adapter；必须同时重跑 paired sensitivity 和闭环 swap。

## 关键文件与产物

代码：

- `tdmpc2/common/world_model.py`
- `tdmpc2/models/task_context_adapter.py`
- `tdmpc2/tdmpc2.py`
- `tdmpc2/scripts/task_vec_sensitivity_report.py`
- `scripts/run_01125_online_family_replay_targets.sh`
- `scripts/run_01125_online_family_acquire_targets.sh`
- `scripts/run_01125_00256_rawtask_adapter_diagnostic.sh`

重要 logs/reports：

- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_acquire_from_01125/20260618_001734_launcher`
- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_polish_01125_00256/20260619_0135_retention_polish_launcher`
- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_taskctx_adapter_01125_00256/20260622_taskctx_adapter_00256_launcher`
- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/20260622_194546_launcher`
- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/20260624_2115_rawtask_alpha0005_sitelimited_launcher`
- `logs/isaaclab-srsa-assembly/1/srsa_axial_online_family_rawtask_adapter_01125_00256/20260625_0010_rawtask_alpha001_sitelimited_launcher`
- `logs/task_vec_swap_eval/20260624_2220_sitelimited_swap_eval`
- `logs/task_vec_swap_eval/20260625_alpha001_sitelimited_swap_eval`
- `logs/task_vec_sensitivity/20260619_00256_v2_offline_report.json`
- `logs/task_vec_sensitivity/20260619_00256_retention_polish_report.json`
- `logs/task_vec_sensitivity/20260622_taskctx_adapter_00256_report.json`
- `logs/task_vec_sensitivity/20260622_rawtask_adapter_00256_report.json`
- `logs/task_vec_sensitivity/20260624_rawtask_alpha0005_sitelimited_00256_report.json`
- `logs/task_vec_sensitivity/20260625_rawtask_alpha001_sitelimited_00256_report.json`
