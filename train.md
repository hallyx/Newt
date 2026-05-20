# Newt / SRSA 训练启动说明

本文档记录当前代码如何启动训练，以及常用参数的意义。

当前主方法已经从 language-conditioned multitask world model 改为：

`structured-task-parameter-conditioned contact-aware axial mating world model`

也就是：

- 保留 Newt / TD-MPC2 的 `state encoder h`、`latent dynamics d`、`reward model R`、`Q model Q`、`policy prior p`
- 保留 demonstration pretraining、online RL fine-tune、demo replay + online replay 混合采样
- 默认旁路 language embedding，不使用 RGB/DINOv2
- 默认使用 `AxialTaskEncoder(task_vec_6) -> c_task(64D)` 作为 task conditioning
- 力传感器/接触信息作为 state observation 的一部分进入 state encoder

## 推荐在线训练命令

从 repo 根目录运行：

```bash
cd /home/gpuserver/hx/github/Newt

/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/train.py \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  assembly_id=01125 \
  srsa_dir=/home/gpuserver/hx/github/srsa \
  srsa_sparse_reward=true \
  srsa_if_sbc=false \
  num_envs=300 \
  gpu_id=0 \
  multiproc=true \
  num_gpus=2 \
  steps=6000000 \
  model_size=S \
  batch_size=1024 \
  buffer_size=1000000 \
  horizon=3 \
  utd=0.075 \
  use_demos=false \
  compile=true \
  enable_wandb=false \
  save_agent=true \
  save_best=true \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=true \
  isaaclab_disable_imitation_reward=true \
  srsa_task_family_name=normal_fit \
  srsa_task_param_obs=false \
  srsa_task_param_obs_mode=task_vec \
  srsa_enable_axial_task_param_sampler=true \
  srsa_axial_fixed_plug_scale=true \
  srsa_axial_clearance_base=0.000114 \
  'srsa_axial_clearance_depth_templates="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"' \
  srsa_axial_clearance_jitter_ratio=0.10 \
  srsa_axial_depth_base=0.015 \
  srsa_axial_depth_jitter_ratio=0.10 \
  'srsa_axial_init_error_xy_range="0.009,0.0010"' \
  'srsa_axial_init_error_z_range="0.0010,0.0020"' \
  'srsa_axial_init_error_yaw_range="-0.0872665,0.0872665"' \
  'srsa_axial_visual_noise_xy_range="0.0,0.001"' \
  'srsa_axial_visual_noise_z_range="0.0,0.0005"' \
  srsa_enable_flange_force_sensor=true \
  isaaclab_canonical_append_force=true \
  isaaclab_canonical_append_task_params=false \
  srsa_vision_noise_xy_std=0.003 \
  srsa_vision_noise_xy_jitter_std=0.0003 \
  isaaclab_canonical_use_visual_noise=true \
  task_conditioning=axial_params \
  progress_log_interval_sec=30 \
  skip_initial_eval=true \
  eval_episodes=1 \
  eval_freq=500000 \
  exp_name=srsa_axial_online
```

如果只是确认环境和输入输出是否正常，可以先跑小步数：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/train.py \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  assembly_id=00783 \
  srsa_sparse_reward=true \
  srsa_if_sbc=false \
  num_envs=1 \
  steps=1 \
  eval_episodes=1 \
  model_size=S \
  batch_size=8 \
  buffer_size=10000 \
  horizon=2 \
  use_demos=false \
  enable_wandb=false \
  save_agent=false \
  save_best=false \
  compile=false \
  mpc=false \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=true \
  isaaclab_disable_imitation_reward=true \
  srsa_task_family_name=normal_fit \
  srsa_task_param_obs=false \
  srsa_task_param_obs_mode=task_vec \
  srsa_enable_axial_task_param_sampler=true \
  srsa_axial_fixed_plug_scale=true \
  srsa_axial_clearance_base=0.000114 \
  'srsa_axial_clearance_depth_templates="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"' \
  srsa_axial_clearance_jitter_ratio=0.10 \
  srsa_axial_depth_base=0.015 \
  srsa_axial_depth_jitter_ratio=0.10 \
  'srsa_axial_init_error_xy_range="0.005,0.0010"' \
  'srsa_axial_init_error_z_range="0.00,0.005"' \
  'srsa_axial_init_error_yaw_range="-0.15,0.15"' \
  'srsa_axial_visual_noise_xy_range="0.0,0.001"' \
  'srsa_axial_visual_noise_z_range="0.0,0.0005"' \
  srsa_enable_flange_force_sensor=true \
  isaaclab_canonical_append_force=true \
  isaaclab_canonical_append_task_params=false \
  task_conditioning=axial_params \
  progress_log_interval_sec=10 \
  skip_initial_eval=true \
  exp_name=srsa_smoke
```

## Debug 输入输出

打开 wrapper 级别的 I/O 检查：

```bash
isaaclab_debug_io=true \
isaaclab_debug_io_steps=3 \
isaaclab_debug_io_every=1
```

会打印：

- raw IsaacLab `policy` observation
- Newt 实际使用的 `output.obs`
- action / reward / done / success
- canonical obs 分段
- force sensor 张量
- SRSA runtime task params
- vision noise 张量

主方法推荐的 observation 是：

- `14D` canonical state
- `+3D` force observation
- 总计 `17D`

任务参数不建议再拼进 observation，主方法使用 `AxialTaskEncoder` 处理任务参数：

```bash
isaaclab_canonical_append_task_params=false
task_conditioning=axial_params
```

## 参数说明

### 环境选择

`isaaclab_backend=srsa`

启用 SRSA 后端。会自动把默认 AutoMate env 切到 SRSA 注册的 `Assembly-Direct-v0`，并导入 `SRSA.tasks`。

`task=isaaclab-srsa-assembly`

Newt 侧的任务名。用于日志目录和 synthetic task metadata。

`assembly_id=00783`

当前装配件 ID。会传给 SRSA runtime，决定加载哪个 plug/socket 资产。

`srsa_dir=/home/gpuserver/hx/github/srsa`

SRSA 仓库路径。wrapper 会把 `source/SRSA` 和 `rl_games_sil` 加进 `sys.path`。

`srsa_sparse_reward=true`

对应 SRSA launcher 的 `--sparse`。默认会把 SRSA env id 从 `Assembly-Direct-v0` 切到 `Assembly-Sparse-v0`。

`srsa_if_sbc=false`

对应 SRSA launcher 的 `--no_sbc`，关闭 SBC curriculum。

`num_envs`

并行环境数量。训练吞吐主要由它决定。例如 `num_envs=200` 时，一轮 74 step episode 会推进 `14800` 个 global steps。

`gpu_id`

训练和仿真使用的起始 GPU。

### 观测与传感器

`isaaclab_use_canonical_obs=true`

使用 Newt wrapper 构造的 canonical state，而不是 IsaacLab 原始 policy obs。

基础 canonical state 是 `14D`：

- `tcp_pos_socket[3]`
- `tcp_quat_socket[4]`
- `tcp_linvel_socket[3]`
- `tcp_angvel_socket[3]`
- `gripper_width[1]`

`srsa_enable_flange_force_sensor=true`

启用 SRSA 侧的法兰/手爪力传感器。

`isaaclab_canonical_append_force=true`

把 `flange_force_obs[3]` 拼进 canonical obs。推荐打开，主方法为 contact-aware world model。

`srsa_flange_force_sensor_body_name=panda_hand`

力传感器挂载的 body 名称。

`srsa_flange_force_sensor_source=held_sensor`

力数据来源。当前推荐保持默认。

`srsa_flange_force_sensor_obs_frame=socket`

力观测表达坐标系。推荐 socket frame，和 canonical state 保持一致。

`srsa_flange_force_sensor_obs_scale=50.0`

力归一化比例。`flange_force_obs = force / scale`。

`isaaclab_canonical_append_task_params=false`

不要把任务参数拼到 observation。主方法的任务参数通过 AxialTaskEncoder 进入模型。

`isaaclab_canonical_append_task_params=true`

仅建议用于 debug 或 ablation。打开后 observation 会按 `srsa_task_param_obs_mode` 额外拼任务参数：

- `task_vec`: 6 维 Newt/SRSA 共享 task vector
- `legacy`: 9 维 legacy task param tensor

### 视觉误差

`srsa_vision_noise_xy_std`

每个 episode 的 XY 视觉定位误差标准差，单位是米。比如 3mm 应写成 `0.003`，不要写成 `3`。

`srsa_vision_noise_xy_jitter_std`

每步 jitter 的 XY 标准差，单位是米。

`srsa_vision_noise_z_std`

Z 方向视觉定位误差标准差，单位是米。

`srsa_vision_noise_z_jitter_std`

每步 jitter 的 Z 标准差，单位是米。

`isaaclab_canonical_use_visual_noise=true`

canonical obs 构造时用带视觉误差的 socket frame。注意：visual noise 不进入 AxialTaskEncoder，只影响 state observation。

### 任务参数与 AxialTaskEncoder

`task_conditioning=axial_params`

默认主方法。模型使用结构化任务参数，不使用 language embedding。

AxialTaskEncoder 输入 `task_vec_6`：

```text
[
  task_type_id_float,
  log_scale,
  clearance_abs_norm,
  clearance_rel_norm,
  depth_abs_norm,
  yaw_requirement_float
]
```

含义：

- `task_type_id_float`: `0=peg_in_hole / 轴装孔`，`1=sleeve_on_shaft / 孔装轴`
- `log_scale`: `log(scale_ratio)`
- `clearance_abs_norm`: `radial_clearance / reference_radius`
- `clearance_rel_norm`: `radial_clearance / male_radius`
- `depth_abs_norm`: `target_insertion_depth / reference_depth`
- `yaw_requirement_float`: `0 or 1`

明确不进入 AxialTaskEncoder 的量：

- `initial_xy_error`
- `visual_noise_std`
- `task_id`
- `assembly_id`

`srsa_task_family_name=normal_fit`

SRSA 内置 fit family。当前可用常见值：

- `normal_fit`
- `loose_fit`
- `tight_fit`
- `baseline`

wrapper 会优先读取 SRSA runtime 的 `current_task_vec`；如果旧环境没有该字段，再从 `current_task_params` 生成当前任务的 `task_vec_6`。在线训练时 replay buffer 也会保存每个 episode 实际采样到的 `current_task_vec`，因此 reset-time sampler 产生的连续任务参数会真正进入 AxialTaskEncoder。

### SRSA reset-time sampler

新 SRSA 训练参数通过 `srsa_*` Hydra 参数透传为 `SRSA_*` 环境变量。当前推荐配置与 `/home/gpuserver/hx/github/srsa/train.sh` 对齐：

```bash
srsa_enable_axial_task_param_sampler=true
srsa_task_param_obs_mode=task_vec
srsa_axial_fixed_plug_scale=true
srsa_axial_clearance_base=0.000114
'srsa_axial_clearance_depth_templates="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"'
srsa_axial_clearance_jitter_ratio=0.10
srsa_axial_depth_base=0.015
srsa_axial_depth_jitter_ratio=0.10
'srsa_axial_init_error_xy_range="0.005,0.0010"'
'srsa_axial_init_error_z_range="0.00,0.005"'
'srsa_axial_init_error_yaw_range="-0.15,0.15"'
'srsa_axial_visual_noise_xy_range="0.0,0.001"'
'srsa_axial_visual_noise_z_range="0.0,0.0005"'
```

联合模板格式是：

```text
"gamma_c:gamma_H;gamma_c:gamma_H;..."
```

其中 `gamma_c` 乘到 `srsa_axial_clearance_base`，`gamma_H` 乘到 `srsa_axial_depth_base`。jitter 会在每个模板附近做连续扰动。

带逗号或分号的 Hydra override 推荐用上面这种单引号包住整段、双引号包住值的写法，避免被 Hydra 解析成 sweep。

也可以显式覆盖几何参数：

```bash
srsa_plug_diameter=0.007986
srsa_hole_diameter=0.008100
srsa_clearance=0.000114
srsa_clearance_ratio=0.014275
srsa_insertion_depth=0.015
```

也可以直接传 6 维向量：

```bash
axial_task_vec_6="[0,0,0.014275,0.014275,1,0]"
```

`task_conditioning=id_embedding`

保留的 ablation。使用旧的 task-id/language embedding 风格，不是主方法。

`task_conditioning=none`

不使用任何 task conditioning，用于 ablation。

### TD-MPC2 训练参数

`steps`

总环境步数。多环境下每次 env step 增加 `num_envs * world_size`。

`model_size=S`

模型规模。常用 `S/B/L/XL`。调试推荐 `S`。

`batch_size`

replay sample batch size。

`buffer_size`

online replay buffer 容量。

`horizon`

world model rollout horizon。常用 `2` 或 `3`。

`utd`

update-to-data ratio。每收集一步数据累计多少 update token。

`seeding_coef`

开始使用策略/更新前，需要先收集多少个 update frequency 的随机数据。

`mpc=true`

启用 TD-MPC2 planning。调试环境时可设为 `false`，会直接用 policy prior。

`compile=false`

关闭 `torch.compile`。调试阶段推荐关闭，稳定后可以再尝试打开。

### eval 与进度输出

`eval_freq`

每隔多少 global steps 做一次 eval。设置 `eval_freq=0` 可暂时关闭周期 eval。

`eval_episodes`

每个 env 至少完成多少个 eval episode。

`skip_initial_eval=true`

跳过第 0 step 的初始 eval，直接开始 rollout。排查“启动后很久没输出”时推荐打开。

`progress_log_interval_sec=30`

心跳打印间隔。即使还没到完整 train log，也会打印当前处于 `eval`、`rollout` 还是 `update`。

`eval_hang_guard_factor=2.0`

eval 卡死保护。超过 `eval_episodes * episode_length * factor` 个 env step 还没完成 eval，会报错提示是否没有返回 `truncated/final_info`。

### 日志与保存

`exp_name`

实验名。日志路径中会包含它。

日志目录格式：

```text
logs/<task>/<seed>/<exp_name>/<run_id>/
```

`enable_wandb=false`

关闭 wandb，只写本地日志。

`save_agent=true`

保存 checkpoint。

`save_best=true`

根据 `save_best_metric` 保存 best checkpoint。

`save_best_metric=episode_success`

默认按 eval success 选 best。

## 离线预训练入口

已有 compact offline dataset：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/offline_train.py \
  offline_dataset_fp=/path/to/compact.pt \
  task=isaaclab-srsa-assembly \
  task_conditioning=axial_params \
  model_size=S \
  batch_size=1024 \
  horizon=3 \
  offline_bc_steps=50000 \
  offline_wm_steps=100000 \
  enable_wandb=false \
  exp_name=offline_axial
```

从 manifest 自动导出并训练：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/offline_train.py \
  offline_manifest_fp=data/offline_manifest_rollout_out.json \
  offline_export_overwrite=false \
  task=isaaclab-srsa-assembly \
  task_conditioning=axial_params \
  model_size=S \
  batch_size=1024 \
  horizon=3 \
  enable_wandb=false \
  exp_name=offline_manifest_axial
```

manifest 中如果提供 `task_vec_6` 或几何字段，会用于构造每个 task 的 axial task vector。

## 用 00186 checkpoint 迁移测试其他 assembly id 并采集 offline RL 数据

新增入口：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/collect_eval_rollouts.py \
  checkpoint=logs/isaaclab-automate-assembly/1/default/20260516_225547_asm-00186/models/best.pt \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  srsa_dir=/home/gpuserver/hx/github/srsa \
  num_envs=200 \
  gpu_id=0 \
  model_size=L \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=false \
  isaaclab_disable_imitation_reward=true \
  srsa_task_family_name=normal_fit \
  srsa_enable_flange_force_sensor=false \
  isaaclab_canonical_append_force=false \
  isaaclab_canonical_append_task_params=false \
  isaaclab_canonical_use_visual_noise=false \
  task_conditioning=id_embedding \
  collect_assembly_ids="[00141,00211,00426,00638,00783]" \
  collect_episodes_per_task=500 \
  collect_source_assembly_id=00186 \
  collect_match_checkpoint=true \
  collect_spawn_per_assembly=true \
  collect_overwrite=true \
  enable_wandb=false \
  exp_name=srsa_00186_direct_eval_collect
```

如果 checkpoint 是旧的 `isaaclab-automate-assembly` 模型，脚本会在
`collect_match_checkpoint=true` 时自动从 checkpoint 推断并覆盖：

- `model_size=L`
- `task_conditioning=id_embedding`
- `isaaclab_use_canonical_obs=false`
- checkpoint 需要的 raw policy obs 维度，当前为 `24D`

多目标 id 采集默认使用 `collect_spawn_per_assembly=true`，每个
`assembly_id` 会单独启动一个 Python/IsaacSim 进程，避免 IsaacSim 在同一进程里反复销毁和重建环境时卡住。

也可以直接从已有 manifest 读取目标 id；脚本会自动排除 `collect_source_assembly_id=00186`：

```bash
offline_manifest_fp=data/offline_manifest_rollout_out.json
```

每个目标 id 会保存：

```text
logs/<task>/<seed>/<exp_name>/<run_id>/data/policy_rollouts_from_00186/<assembly_id>/policy_eval_rollouts.pt
logs/<task>/<seed>/<exp_name>/<run_id>/data/policy_rollouts_from_00186/offline_manifest_eval_rollouts.json
```

下一阶段 offline RL 可以直接使用采集出的 manifest：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/offline_train.py \
  offline_manifest_fp=/path/to/offline_manifest_eval_rollouts.json \
  offline_export_overwrite=true \
  task=isaaclab-srsa-assembly \
  task_conditioning=axial_params \
  model_size=S \
  batch_size=1024 \
  horizon=3 \
  enable_wandb=false \
  exp_name=offline_from_00186_policy_rollouts
```

如果要用当前采集数据继续微调旧 `isaaclab-automate-assembly` Newt checkpoint，使用：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/offline_train.py \
  checkpoint=logs/isaaclab-automate-assembly/1/default/20260516_225547_asm-00186/models/best.pt \
  offline_manifest_fp=data/offline_manifest_policy_rollouts_from_00186.json \
  offline_export_fp=data/offline_policy_rollouts_from_00186_compact.pt \
  offline_export_overwrite=false \
  task=isaaclab-srsa-assembly \
  model_size=L \
  task_conditioning=id_embedding \
  learn_task_emb=true \
  batch_size=512 \
  horizon=3 \
  compile=false \
  gpu_id=0 \
  offline_gpu_id=0 \
  offline_bc_steps=10000 \
  offline_wm_steps=50000 \
  offline_rl_steps=50000 \
  offline_log_freq=200 \
  offline_save_freq=5000 \
  enable_wandb=false \
  save_agent=true \
  exp_name=offline_rl_from_00186_policy_rollouts
```

这个流程会从旧 checkpoint 继续：

- `bc`: 用采集动作做行为克隆 warm-up
- `wm`: 更新 latent dynamics / reward / Q，同时保留 BC prior
- `rl`: 打开 Max-Q policy update，进行离线 RL 微调

当前已合成的数据入口：

```text
data/offline_manifest_policy_rollouts_from_00186.json
data/offline_policy_rollouts_from_00186_compact.pt
```

离线 RL 微调完成后，对 manifest 中不同任务批量测试：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python tdmpc2/batch_eval_tasks.py \
  checkpoint=logs/isaaclab-srsa-assembly/1/offline_rl_from_00186_policy_rollouts/<run_id>/models/final.pt \
  offline_manifest_fp=data/offline_manifest_policy_rollouts_from_00186.json \
  isaaclab_backend=srsa \
  task=isaaclab-srsa-assembly \
  srsa_dir=/home/gpuserver/hx/github/srsa \
  num_envs=200 \
  gpu_id=0 \
  model_size=L \
  horizon=3 \
  compile=false \
  mpc=true \
  isaaclab_headless=true \
  isaaclab_use_canonical_obs=false \
  isaaclab_disable_imitation_reward=true \
  srsa_task_family_name=normal_fit \
  srsa_enable_flange_force_sensor=false \
  isaaclab_canonical_append_force=false \
  isaaclab_canonical_append_task_params=false \
  isaaclab_canonical_use_visual_noise=false \
  task_conditioning=id_embedding \
  learn_task_emb=true \
  batch_eval_episodes_per_task=100 \
  batch_eval_spawn_per_assembly=true \
  batch_eval_overwrite=true \
  enable_wandb=false \
  exp_name=batch_eval_offline_rl_from_00186
```

如果只想测其中几个任务，加：

```bash
batch_eval_assembly_ids="[00141,00211]"
```

结果会保存为：

```text
logs/isaaclab-srsa-assembly/1/batch_eval_offline_rl_from_00186/<run_id>/batch_eval/<checkpoint_name>/batch_eval_summary.json
logs/isaaclab-srsa-assembly/1/batch_eval_offline_rl_from_00186/<run_id>/batch_eval/<checkpoint_name>/batch_eval_summary.csv
```

## 常见现象

### 训练启动后一直没看到 train/eval 行

先看是否出现 heartbeat：

```text
eval     progress ...
rollout  progress ...
update   progress ...
```

如果没有 heartbeat，确认命令里有：

```bash
progress_log_interval_sec=10
```

如果卡在 eval，可以临时跳过初始 eval：

```bash
skip_initial_eval=true eval_episodes=1
```

如果想先完全不 eval：

```bash
skip_initial_eval=true eval_freq=0
```

### Q-functions 的输入维度怎么看

新 axial 主方法、`model_size=S`、canonical `17D` observation 时，通常会看到：

```text
Axial task encoder ... -> 64D
Q-functions ... in_features=454
```

其中：

```text
454 = latent_dim 384 + action_dim 6 + task_dim 64
```

如果看到类似：

```text
Q-functions ... in_features=902
```

通常说明还在走旧的 `task_dim=512` embedding 路径，检查是否误设了：

```bash
task_conditioning=id_embedding
```

主方法应使用：

```bash
task_conditioning=axial_params
```

### `nvcc command not found`

如果设置了：

```bash
isaaclab_disable_imitation_reward=true
```

通常不会阻断训练。它主要和旧 AutoMate imitation reward / SoftDTW 相关。

### resource_tracker leaked semaphore

Isaac Sim 退出时常见 warning，一般不是训练失败原因。

### 命令参数被 shell 当成重定向

Hydra 参数里如果含有特殊字符，建议用引号。尤其是：

```bash
axial_task_vec_6="[0,0,0.014275,0.014275,1,0]"
```
