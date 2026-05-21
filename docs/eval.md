# Newt / SRSA Eval 使用说明

本文档集中记录 `tdmpc2/eval.py` 和 `tdmpc2/batch_eval_tasks.py` 的评测方式、关键参数和常用命令。

当前 eval 分两类：

- `eval_mode=sim`: 仿真闭环评测。observation、action、reward、success 都来自 Isaac/SRSA 仿真环境。
- `eval_mode=real eval_real_mode=stream`: 真机 action 发送烟测。Newt 仍创建一个 SRSA/Isaac 环境用于策略输入和流程控制，同时把选中 env 的 6D action 通过 ZMQ 发给真机侧 receiver。
- `eval_mode=real eval_real_mode=closed_loop`: 真机 observation 闭环推理。真机侧发送最新 canonical observation，Newt 推理 6D action，再通过 ZMQ 发给机械臂控制端。

注意：`stream` 只适合检查动作链路，不是真机 observation 闭环。真机 success 需要看真机侧日志或外部记录。

## 入口选择

### 单任务 Eval

使用：

```bash
tdmpc2/eval.py
```

适合：

- 测一个 checkpoint 在一个 `task_id` 或一个 assembly 上的表现
- 检查 `task_id -> SRSA 参数 -> task_vec_6` 是否一致
- 真机 ZMQ action smoke test

### 批量 Eval

使用：

```bash
tdmpc2/batch_eval_tasks.py
```

适合：

- 对 manifest 中多个 `task_id / assembly_id` 批量评测
- 输出 JSON 和 CSV 汇总
- 多 assembly 使用独立子进程，减少 IsaacSim 重复销毁环境导致的卡住风险

## Checkpoint 与 Observation 必须匹配

eval 命令中的 observation 配置必须和 checkpoint 训练时一致。

### 14D canonical checkpoint

基础 canonical state：

```text
tcp_pos_socket[3]
tcp_quat_socket[4]
tcp_linvel_socket[3]
tcp_angvel_socket[3]
gripper_width[1]
```

关键参数：

```bash
isaaclab_use_canonical_obs=true
srsa_enable_flange_force_sensor=false
isaaclab_canonical_append_force=false
isaaclab_canonical_append_task_params=false
task_conditioning=axial_params
```

debug 时应看到：

```text
obs_space=(14,)
```

### 17D contact-aware checkpoint

在 14D 后拼接 `flange_force_obs[3]`。

关键参数：

```bash
isaaclab_use_canonical_obs=true
srsa_enable_flange_force_sensor=true
isaaclab_canonical_append_force=true
isaaclab_canonical_append_task_params=false
task_conditioning=axial_params
```

debug 时应看到：

```text
obs_space=(17,)
runtime.flange_force_obs: shape=(num_envs, 3)
```

### ContactHistoryEncoder checkpoint

如果 checkpoint 使用了：

```bash
contact_history_enabled=true
```

模型侧 latent dynamics 会额外需要 `contact_context`：

```text
force_history:    (B, H, 6)
action_history:   (B, H, 6)
ee_delta_history: (B, H, 6)  # optional

ContactHistoryEncoder -> contact_context: (B, contact_context_dim)
```

当前实现中，`WorldModel.next()` 已支持传入这些 history；如果 eval 主循环没有传 history，会使用零 `contact_context`。这保证 checkpoint 能跑通，但不会发挥接触历史的作用。

真机闭环第一版可以不改 SRSA 环境：

- `action_history` 由 Newt 记录过去发给机械臂的 6D action。
- `ee_delta_history` 可由连续真机 canonical obs 的末端 pose 差分得到。
- `force_history` 若暂时只有 3D 力，可填成 `[Fx,Fy,Fz,0,0,0]`。

如果真机或 SRSA 侧能提供 6D wrench，推荐 observation/history 侧使用：

```text
[Fx,Fy,Fz,Tx,Ty,Tz] in socket frame
```

并保持和训练时相同的归一化比例。

### 旧 raw policy obs / id embedding checkpoint

旧 `isaaclab-automate-assembly` checkpoint 常见配置：

```bash
isaaclab_use_canonical_obs=false
srsa_enable_flange_force_sensor=false
isaaclab_canonical_append_force=false
isaaclab_canonical_append_task_params=false
task_conditioning=id_embedding
learn_task_emb=true
```

debug 时常见：

```text
raw.policy_obs: shape=(num_envs, 24)
```

如果 checkpoint 加载失败，优先检查：

- `model_size`
- `task_conditioning`
- observation 维度
- `action_dim`
- `num_global_tasks / eval_task_id`

## SRSA task id + 参数模板

仓库内置模板：

```text
data/srsa_axial_task_templates.json
```

这个文件现在只保存 clearance/depth 参数模板；真实几何从 SRSA 导出的 mesh CSV 读取：

```text
/home/gpuserver/hx/github/srsa/outputs/mesh_geometry_params/srsa_mesh_geometry_params.csv
```

推荐用法是：

- `assembly_id=00186`: 选择 SRSA mesh/task id。
- `srsa_task_template_id=3`: 选择参数模板，例如 `c2.0-d1.5`。
- `srsa_task_template_fp=data/srsa_axial_task_templates.json`: 让 Newt 自动把二者合成 SRSA 参数和 `task_vec_6`。

```bash
assembly_id=00186
srsa_task_template_fp=data/srsa_axial_task_templates.json
srsa_task_template_id=3
```

也可以用 `srsa_mesh_geometry_task_id=00186` 显式指定 mesh id；没有设置时默认使用 `assembly_id`。为了避免命名混淆，参数模板也可以写成 `srsa_param_template_id=3`，代码会自动映射到 `srsa_task_template_id/eval_task_id`。

也可以从已有 manifest 按 id 选择：

```bash
offline_manifest_fp=/path/to/offline_manifest_eval_rollouts.json
eval_task_id=3
```

关键参数：

```bash
eval_task_template_exact=true
eval_task_template_print=true
```

含义：

- `eval_task_template_exact=true`: 将 `assembly_id + 参数模板` 展开的 `task_vec_6` 解码成固定 SRSA sampler ranges，使环境侧几何/深度参数和模型侧 `task_vec_6` 一致。
- `eval_task_template_print=true`: 启动时打印被应用的 `task_id / assembly_id / task_vec_6`。

默认 mesh 列映射：

```text
plug_diameter      <- plug_xy_bbox_max
mesh_hole_diameter <- socket_xy_bbox_max
clearance_base     <- 2 * plug_to_socket_surface_dist_p05
depth_base         <- plug_bbox_z
reference_radius   <- plug_xy_radius_p95_from_centroid
reference_depth    <- plug_bbox_z
```

其中 `srsa_task_template_id` 的 `clearance_multiplier/depth_multiplier` 会分别乘到 `clearance_base/depth_base`。如果要换 proxy，可以覆盖：

```bash
srsa_mesh_clearance_column=xy_bbox_max_clearance_proxy
srsa_mesh_clearance_mode=diametral
srsa_mesh_depth_column=plug_bbox_z
```

验证时重点看 debug 输出：

```text
runtime.task_vec
runtime.task_param_tensor
runtime.current_task_params
```

例如 `assembly_id=00186 srsa_task_template_id=3` 解析阶段会打印类似：

```text
Applied eval task template: task_id=3 assembly_id=00186 task_vec_6=[0, 1.23926, 0.237008, 0.238145, 1.5, 0]
```

## 仿真 Eval 命令

### 14D canonical 仿真 Eval

```bash
cd /home/gpuserver/hx/github/Newt

/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/eval.py \
  checkpoint=/path/to/checkpoint.pt \
  eval_mode=sim \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  srsa_dir=/home/gpuserver/hx/github/srsa \
  assembly_id=00186 \
  srsa_task_template_fp=data/srsa_axial_task_templates.json \
  srsa_task_template_id=3 \
  num_envs=1 \
  eval_trials=10 \
  model_size=S \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=true \
  srsa_enable_flange_force_sensor=false \
  isaaclab_canonical_append_force=false \
  isaaclab_canonical_append_task_params=false \
  task_conditioning=axial_params \
  enable_wandb=false \
  exp_name=eval_sim_14d
```

### 17D contact-aware 仿真 Eval

```bash
cd /home/gpuserver/hx/github/Newt

/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/eval.py \
  checkpoint=logs/isaaclab-srsa-assembly/1/srsa_axial_online/20260521_105015_asm-01125/models/best.pt \
  eval_mode=sim \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  srsa_dir=/home/gpuserver/hx/github/srsa \
  assembly_id=01125 \
  srsa_task_template_fp=data/srsa_axial_task_templates.json \
  srsa_task_template_id=3 \
  num_envs=100 \
  eval_trials=500 \
  model_size=S \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=true \
  srsa_enable_flange_force_sensor=true \
  isaaclab_canonical_append_force=true \
  isaaclab_canonical_append_task_params=false \
  task_conditioning=axial_params \
  enable_wandb=false \
  exp_name=eval_sim_17d
```

### 带 Debug 输出的 Eval

用于验证参数和 observation 维度：

```bash
isaaclab_debug_io=true \
isaaclab_debug_io_steps=1 \
isaaclab_debug_io_every=1
```

完整示例：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/eval.py \
  checkpoint=/path/to/checkpoint.pt \
  eval_mode=sim \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  assembly_id=00186 \
  srsa_task_template_fp=data/srsa_axial_task_templates.json \
  srsa_task_template_id=3 \
  num_envs=1 \
  eval_trials=1 \
  model_size=S \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=true \
  task_conditioning=axial_params \
  isaaclab_debug_io=true \
  isaaclab_debug_io_steps=1 \
  isaaclab_debug_io_every=1 \
  enable_wandb=false \
  exp_name=eval_debug
```

## 真机 Eval 命令

真机侧需要先启动 action receiver，并监听与 `eval_zmq_server` 对应的地址。

Newt 发送消息格式由 `tdmpc2/zmq_action_publisher.py` 定义，核心字段：

```text
delta: 6D action
action: 6D action
task_id
episode_step
done
source: newt_eval
action_frame
action_order
```

### 真机闭环控制

最新 17D contact-aware checkpoint 应优先用这个模式。真机侧每个控制周期发送一条 observation JSON：

```json
{
  "obs": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
  "task_vec_6": [0.0, 0.0, 0.014275, 0.014275, 1.0, 0.0],
  "episode_step": 0,
  "done": false
}
```

`obs` 必须和 checkpoint 训练 observation 维度一致：

- 14D: `tcp_pos_socket[3] + tcp_quat_wxyz[4] + tcp_linvel_socket[3] + tcp_angvel_socket[3] + gripper_width[1]`
- 17D: 上面 14D 再拼 `flange_force_obs[3]`

启动命令：

```bash
cd /home/gpuserver/hx/github/Newt

/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/eval.py \
  checkpoint=/path/to/checkpoint.pt \
  eval_mode=real \
  eval_real_mode=closed_loop \
  eval_real_obs_server=tcp://<robot-host>:5556 \
  eval_real_obs_connect=true \
  eval_real_obs_timeout_ms=1000 \
  eval_real_steps=74 \
  eval_zmq_server=tcp://<robot-host>:5555 \
  eval_zmq_rate=10 \
  eval_zmq_action_scale=0.05 \
  eval_zmq_action_frame=socket \
  'eval_zmq_action_order="dx,dy,dz,droll,dpitch,dyaw"' \
  eval_zmq_send_done=true \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  srsa_task_template_fp=data/srsa_axial_task_templates.json \
  srsa_task_template_id=2 \
  num_envs=1 \
  model_size=S \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_use_canonical_obs=true \
  srsa_enable_flange_force_sensor=true \
  isaaclab_canonical_append_force=true \
  isaaclab_canonical_append_task_params=false \
  task_conditioning=axial_params \
  enable_wandb=false \
  exp_name=eval_real_closed_loop
```

控制机械臂时，机器人侧应把收到的 `delta/action` 当作归一化 6D 增量，按 `eval_zmq_action_scale` 和自身安全限幅映射到末端控制命令。不要让 Newt 直接绕过机器人侧的速度、位移、力、碰撞和 workspace 限幅。

### 真机低速 Smoke Test

这个模式保留用于检查 action receiver 和方向，不使用真机 observation 闭环。首次建议使用很小动作尺度：

```bash
cd /home/gpuserver/hx/github/Newt

/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/eval.py \
  checkpoint=/path/to/checkpoint.pt \
  eval_mode=real \
  eval_real_mode=stream \
  eval_zmq_server=tcp://<robot-host>:5555 \
  eval_zmq_env_index=0 \
  eval_zmq_rate=5 \
  eval_zmq_action_scale=0.05 \
  eval_zmq_send_done=true \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  srsa_dir=/home/gpuserver/hx/github/srsa \
  assembly_id=00186 \
  srsa_task_template_fp=data/srsa_axial_task_templates.json \
  srsa_task_template_id=3 \
  num_envs=1 \
  eval_trials=1 \
  model_size=S \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=true \
  task_conditioning=axial_params \
  enable_wandb=false \
  exp_name=eval_real_smoke
```

### 真机常规 Action Streaming

确认方向和限幅后，可以逐步提高动作尺度和频率：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/eval.py \
  checkpoint=/path/to/checkpoint.pt \
  eval_mode=real \
  eval_real_mode=stream \
  eval_zmq_server=tcp://<robot-host>:5555 \
  eval_zmq_env_index=0 \
  eval_zmq_rate=10 \
  eval_zmq_action_scale=0.10 \
  eval_zmq_send_done=true \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  assembly_id=00186 \
  srsa_task_template_fp=data/srsa_axial_task_templates.json \
  srsa_task_template_id=3 \
  num_envs=1 \
  eval_trials=1 \
  model_size=S \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=true \
  task_conditioning=axial_params \
  enable_wandb=false \
  exp_name=eval_real_zmq
```

真机安全建议：

- 第一次使用 `eval_zmq_action_scale=0.05`。
- 第一次使用 `eval_zmq_rate=5`。
- `num_envs=1`，并固定 `eval_zmq_env_index=0`。
- 确认机器人侧 receiver 做了速度、位移、力和工作空间限幅。
- 真机侧必须能处理 `done=true` 的零 action 消息。
- 不要用仿真 `episode_success` 判断真机是否成功。

## Batch Eval 命令

批量 eval 必须提供 manifest：

```bash
offline_manifest_fp=data/offline_manifest_policy_rollouts_from_00186.json
```

示例：

```bash
cd /home/gpuserver/hx/github/Newt

/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/batch_eval_tasks.py \
  checkpoint=/path/to/checkpoint.pt \
  offline_manifest_fp=data/offline_manifest_policy_rollouts_from_00186.json \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  srsa_dir=/home/gpuserver/hx/github/srsa \
  num_envs=100 \
  gpu_id=0 \
  model_size=S \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=true \
  task_conditioning=axial_params \
  batch_eval_episodes_per_task=100 \
  batch_eval_spawn_per_assembly=true \
  batch_eval_overwrite=true \
  enable_wandb=false \
  exp_name=batch_eval_sim
```

只测部分 assembly：

```bash
batch_eval_assembly_ids="[00141,00211]"
```

输出路径默认：

```text
logs/<task>/<seed>/<exp_name>/<run_id>/batch_eval/<checkpoint_name>/batch_eval_summary.json
logs/<task>/<seed>/<exp_name>/<run_id>/batch_eval/<checkpoint_name>/batch_eval_summary.csv
```

## 常用参数解释

`checkpoint`

要评测的模型权重。必须与 `model_size`、observation 维度和 `task_conditioning` 匹配。

`eval_mode`

`sim` 或 `real`。`real` 会自动启用 `eval_zmq_enabled=true`。

`eval_real_mode`

真机模式：

- `closed_loop`: 真机发送 canonical obs，Newt 推理并发送 6D action，推荐用于真实控制。
- `stream`: 旧 action streaming 烟测，用仿真 obs 驱动策略，只把 action 转发给真机。

`eval_real_obs_server`

真机 observation ZMQ endpoint。默认 Newt 作为 PULL client 连接这个地址；若希望 Newt bind，设置 `eval_real_obs_connect=false`。

`eval_real_steps`

真机闭环最多推理多少步。默认使用 checkpoint/eval 配置里的 episode length。

`eval_trials`

精确评测多少个完成 episode。推荐 eval 脚本使用这个参数，尤其是 `num_envs > 1` 时。

`eval_episodes`

`eval_trials` 未设置时使用。含义是每个 env 至少完成多少个 episode。

`eval_task_id`

从 `offline_manifest_fp` 中选择某个任务。使用 `srsa_task_template_fp` 时，它会自动等于 `srsa_task_template_id`，表示模型侧第几个参数模板。

`assembly_id`

SRSA mesh/task id，例如 `00186`。使用当前模板文件时，Newt 会用它去 mesh CSV 中查找几何 proxy。

`srsa_task_template_fp`

SRSA 参数模板 JSON。用于没有 offline manifest 时按 `assembly_id + srsa_task_template_id` 生成 SRSA 参数方案。

`srsa_task_template_id`

选择 `srsa_task_template_fp` 中的参数模板 id，例如 `3 -> c2.0-d1.5`。解析时也会同步设置 `eval_task_id`。

`srsa_param_template_id`

`srsa_task_template_id` 的清晰别名。推荐在新命令里用它表达“参数模板 id”，把 `assembly_id=00186` 留给 SRSA mesh/task id。

`srsa_mesh_geometry_fp`

可选。覆盖模板文件里的 mesh CSV 路径。默认使用 SRSA 侧 `outputs/mesh_geometry_params/srsa_mesh_geometry_params.csv`。

`eval_task_template_exact`

为 `true` 时，将 `assembly_id + 参数模板` 得到的 `task_vec_6` 解码成固定 SRSA sampler ranges。推荐保持 `true`。

`isaaclab_use_canonical_obs`

是否使用 Newt wrapper 构造的 canonical obs。新方法通常为 `true`。

`isaaclab_canonical_append_force`

是否把 `flange_force_obs[3]` 拼进 observation。只有训练 checkpoint 是 17D 时才打开。

`isaaclab_canonical_append_task_params`

是否把 task params 拼进 observation。主方法一般保持 `false`，因为 task 参数通过 AxialTaskEncoder 进入模型。

`contact_history_enabled`

是否启用模型侧 ContactHistoryEncoder。只有 checkpoint 训练时启用了该模块，eval 才应打开。打开后 dynamics 会多拼 `contact_context`；如果当前 eval 路径没有传 history，会用零 context 兜底。

`contact_history_len`

ContactHistoryEncoder 使用的历史窗口长度 `H`，必须与 checkpoint 一致。

`contact_context_dim`

接触历史编码维度，必须与 checkpoint 一致。

`task_conditioning`

常用值：

- `axial_params`: 主方法，使用 `task_vec_6 -> AxialTaskEncoder`
- `id_embedding`: 旧 ablation 或旧 checkpoint 路径
- `none`: 不使用 task conditioning

`mpc`

是否使用 TD-MPC2 planning。正式 eval 通常为 `true`；debug 动作链路可临时设为 `false`。

`compile`

是否启用 `torch.compile`。排查问题时建议 `false`。

`eval_zmq_server`

真机 receiver 地址，例如：

```text
tcp://192.168.1.10:5555
```

`eval_zmq_rate`

ZMQ action 发送频率。`0` 表示不额外限速。

`eval_zmq_action_scale`

发送给真机前对 6D action 做统一缩放。真机首次测试建议 `0.05`。

`eval_zmq_action_frame`

写入 action JSON 的元信息，默认 `socket`。机器人侧应按这个坐标系解释 6D 增量，或在 receiver 里明确转换。

`eval_zmq_action_order`

写入 action JSON 的元信息，默认：

```text
dx,dy,dz,droll,dpitch,dyaw
```

## 排查 checklist

### 参数模板是否生效

打开：

```bash
isaaclab_debug_io=true
isaaclab_debug_io_steps=1
```

检查：

```text
Applied eval task template: task_id=...
runtime.task_vec
runtime.current_task_params
```

### Observation 维度是否匹配

检查：

```text
obs_space=(14,)
obs_space=(17,)
raw.policy_obs: shape=(..., 24)
```

这些必须和 checkpoint 训练配置一致。

### 真机没有收到 action

检查：

- `eval_mode=real`
- `eval_real_mode=closed_loop` 或 `stream` 是否符合当前真机程序
- `eval_zmq_server` 地址和端口
- 真机 receiver 是否已经启动
- 防火墙和网络是否允许 ZMQ 连接
- 是否安装了 `pyzmq`

### 真机闭环没有收到 observation

检查：

- 真机侧是否向 `eval_real_obs_server` 发送 JSON
- `eval_real_obs_connect` 和真机侧 bind/connect 方向是否匹配
- JSON 中是否有 `obs`，且长度等于 checkpoint observation 维度
- 17D checkpoint 是否补齐 `flange_force_obs[3]`
- `task_vec_6` 是否和当前装配任务一致，或 eval 命令是否正确选择了 `srsa_task_template_id`

### ContactHistoryEncoder 没有效果

检查：

- checkpoint 是否确实用 `contact_history_enabled=true` 训练
- eval/训练 rollout 是否真的传入了 `force_history/action_history/ee_delta_history`
- 当前是否只是走零 `contact_context` 兜底
- 3D force 是否按训练约定扩成 6D，或真机/SRSA 是否提供了同坐标系、同归一化的 6D wrench

### Checkpoint 加载失败

优先核对：

- `model_size`
- `task_conditioning`
- `num_global_tasks`
- `eval_task_id`
- observation 维度
- `action_dim`
