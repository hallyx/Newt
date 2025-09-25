from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from tensordict import from_modules


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"
	

class _VCReg(Function):
	@staticmethod
	def forward(ctx, input, var, cov, epsilon):
		# Save parameters and original input for backward
		ctx.save_for_backward(input)
		ctx.var = var
		ctx.cov = cov
		ctx.epsilon = epsilon
		return input.clone()

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		var = ctx.var
		cov = ctx.cov
		epsilon = ctx.epsilon

		# Flatten to [N, D] where N is batch * time (or any prefix dims), D is feature dim
		flat = input.view(-1, input.size(-1))  # [N, D]
		N, D = flat.shape
		factor = N - 1

		# Compute covariance matrix
		cov_mat = torch.mm(flat.T, flat) / factor  # [D, D]

		# Diagonal (per-dimension std dev) term
		diag = torch.rsqrt(cov_mat.diagonal() + epsilon)
		diag = F.threshold(diag, 1.0, 0.0)  # clamp std gradients

		std_grad = diag * flat  # [N, D]

		# Off-diagonal covariance penalty
		cov_mat_offdiag = cov_mat - torch.diag_embed(torch.diagonal(cov_mat))  # zero diag
		cov_grad = torch.mm(flat, cov_mat_offdiag)  # [N, D]

		# Combine gradients
		grad_input = grad_output \
			- (var / (D * factor)) * std_grad.view_as(grad_output) \
			+ (4 * cov / (D * (D - 1))) * cov_grad.view_as(grad_output)

		return grad_input, None, None, None


class VCReg(nn.Module):
	def __init__(self, var=0.1, cov=0.01, epsilon=1e-5):  # vcreg4
		"""
		VCReg module for variance-covariance regularization.
		Returns identity in forward, adds gradient penalty in backward.
		"""
		super().__init__()
		self.var = var
		self.cov = cov
		self.epsilon = epsilon

	def forward(self, x):
		return _VCReg.apply(x, self.var, self.cov, self.epsilon)

	def extra_repr(self):
		return f"var={self.var}, cov={self.cov}, epsilon={self.epsilon}"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation.
	"""

	def __init__(self, *args, act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act

	def forward(self, x):
		x = super().forward(x)
		return self.act(self.ln(x))

	def __repr__(self):
		if isinstance(self.act, nn.Sequential):
			# print the name of each module in the sequential
			act = '[' + ', '.join([m.__class__.__name__ for m in self.act]) + ']'
		else:
			act = self.act.__class__.__name__
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}, "\
			f"act={act})"


class AdapterLayer(nn.Linear):
	"""
	Adapter layer for stacked visual features.
	Applies the same linear transformation to each feature in the stack,
	and then computes latent difference with the first feature.
	"""

	def __init__(self, *args, task_dim=0, act=None, **kwargs):
		in_dim = (args[0] - task_dim) // 3  # Assumes input is a stack of 3 features
		out_dim = args[1] // 3
		args = (in_dim + task_dim, out_dim,) + args[2:]  # Update in_dim in args
		super().__init__(*args, **kwargs)
		self.task_dim = task_dim
		self.act = act if act is not None else nn.Identity()

	def forward(self, x):
		x_feat, x_task = x[..., :-self.task_dim], x[..., -self.task_dim:]
		x_feat = torch.stack(x_feat.chunk(3, dim=-1), dim=-2)  # Stack features along a new dimension
		x_task = x_task.unsqueeze(-2)  # Add a new dimension for task embedding
		x_task = x_task.expand(*x_task.shape[:-2], 3, self.task_dim)
		x = torch.cat((x_feat, x_task), dim=-1)
		x = super().forward(x)  # Apply linear transformation
		x = self.act(x)
		x = x.view(*x.shape[:-2], -1)  # Flatten the last two dimensions
		# x = x - x[..., 0:1, :]  # Compute difference with the first feature
		return x


def mlp(in_dim, mlp_dims, out_dim, act=None):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1]))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def policy(in_dim, mlp_dims, out_dim, act=None):
	"""
	Policy network for TD-MPC2.
	Vanilla MLP with ReLU activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(nn.Linear(dims[i], dims[i+1]))
		mlp.append(nn.ReLU())
	mlp.append(nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


class QEnsemble(nn.Module):
	"""
	Vectorized ensemble of Q-networks. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		in_dim = cfg.latent_dim + cfg.action_dim + cfg.task_dim
		mlp_dims = 2*[cfg.mlp_dim]
		out_dim = max(cfg.num_bins, 1)
		self._Qs = nn.ModuleList([mlp(in_dim, mlp_dims, out_dim) for _ in range(cfg.num_q)])
		if cfg.compile:
			if cfg.rank == 0:
				print('Compiling QEnsemble forward...')
			self._forward = torch.compile(self._forward_impl, mode='reduce-overhead')
		else:
			self._forward = self._forward_impl
	
	def _forward_impl(self, x):
		outs = [q(x) for q in self._Qs]
		return torch.stack(outs, dim=0)

	def forward(self, x):
		return self._forward(x)


class QOnlineTargetEnsemble(nn.Module):
	"""
	Online and target Q-ensembles for TD-MPC2. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.online = QEnsemble(cfg)
		self.target = deepcopy(self.online)
		self.tau = cfg.tau
		self.target.train(False)
		self.track_grad(False, network='target')

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		self.online.train(mode)
		self.target.train(False)
		return self
	
	def track_grad(self, mode=True, network='online'):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		"""
		assert network in {'online', 'target'}
		module = self.online if network == 'online' else self.target
		for p in module.parameters():
			p.requires_grad_(mode)

	@torch.no_grad()
	def hard_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.copy_(op.data)

	@torch.no_grad()
	def soft_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.lerp_(op.data, self.tau)

	def forward(self, x, target=False):
		if target:
			return self.target(x)
		else:
			return self.online(x)


# def conv(in_shape, num_channels, act=None):
# 	"""
# 	Basic convolutional encoder for TD-MPC2 with raw image observations.
# 	4 layers of convolution with ReLU activations.
# 	"""
# 	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
# 	layers = [
# 		ShiftAug(), PixelPreprocess(),
# 		nn.Conv2d(in_shape[0]+2, num_channels, 7, stride=2), nn.ReLU(inplace=False),
# 		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
# 		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
# 		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]

# 	# # check output shape
# 	# x = torch.randn(1, *in_shape)
# 	# out_pre_flatten = nn.Sequential(*layers[:-1])(x)
# 	# out = nn.Sequential(*layers)(x)
# 	# print('out_pre_flatten shape:', out_pre_flatten.shape)
# 	# print('out shape:', out.shape)
# 	if act:
# 		layers.append(act)
# 	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	if cfg.obs == 'state':
		out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	elif cfg.obs == 'rgb':
		out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim + cfg.obs_shape['rgb'][0], max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	# for k in cfg.obs_shape.keys():
	# 	if k == 'state':
	# 		out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	# 	# elif k == 'rgb':
	# 	# 	out[k] = conv(cfg.obs_shape[k], cfg.num_channels, act=SimNorm(cfg))
	# 	elif k == 'rgb':
	# 		# out[k] = nn.Sequential(
	# 		# 	nn.Linear(cfg.latent_dim + cfg.task_dim + cfg.obs_shape[k][0], cfg.latent_dim),
	# 		# 	SimNorm(cfg)
	# 		# )
	# 		# out[k] = nn.Sequential(
	# 		# 	nn.Linear(cfg.obs_shape[k][0], cfg.latent_dim),
	# 		# 	SimNorm(cfg)
	# 		# )
	# 		# out['fuse'] = nn.Sequential(
	# 		# 	nn.Linear(2*cfg.latent_dim, cfg.latent_dim),
	# 		# 	SimNorm(cfg),
	# 		# )
	# 	else:
	# 		raise NotImplementedError(f"Encoder for observation type {k} not implemented.")
	return nn.ModuleDict(out)


def legacy_api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	encoder_key = 'module._encoder.state.0.weight'
	if encoder_key in source_state_dict and encoder_key not in target_state_dict:
		# Remove 'module.' prefix from all keys in source_state_dict
		source_state_dict = {k[len('module.'):]: v for k, v in source_state_dict.items()}
	if encoder_key in target_state_dict and encoder_key not in source_state_dict:
		# Add 'module.' prefix to all keys in source_state_dict
		source_state_dict = {'module.' + k: v for k, v in source_state_dict.items()}

	for key in ['_encoder.state.0.weight', 'module._encoder.state.0.weight']:
		if key in target_state_dict and key in source_state_dict and \
				target_state_dict[key].shape != source_state_dict[key].shape:
			# rgb input in target but not in source, we should pad
			pad = target_state_dict[key].shape[1] - source_state_dict[key].shape[1]
			assert pad > 0, 'pad should be positive'
			pad_tensor = torch.zeros(source_state_dict[key].shape[0], pad, device=source_state_dict[key].device)
			source_state_dict[key] = torch.cat([source_state_dict[key], pad_tensor], dim=1)

	# rgb_encoder_key = '_encoder.rgb.0.weight'
	# for rgb_key in [rgb_encoder_key, 'module.' + rgb_encoder_key]:
	# 	if rgb_key in target_state_dict and rgb_key not in source_state_dict:
	# 		# Copy rgb encoder weights from target to source
	# 		for key in target_state_dict.keys():
	# 			if key.startswith('_encoder.rgb.') or key.startswith('module._encoder.rgb.'):
	# 				source_state_dict[key] = target_state_dict[key]
	
	if not '_action_masks' in target_state_dict or not '_action_masks' in source_state_dict:
		return source_state_dict

	if '_action_masks' in target_state_dict and '_action_masks' in source_state_dict and \
			source_state_dict['_action_masks'].shape != target_state_dict['_action_masks'].shape:
		# repeat first dimension to match
		source_state_dict['_action_masks'] = source_state_dict['_action_masks'].repeat(
			target_state_dict['_action_masks'].shape[0] // source_state_dict['_action_masks'].shape[0], 1)
		if '_task_emb.weight' in source_state_dict:
			source_state_dict['_task_emb.weight'] = source_state_dict['_task_emb.weight'].repeat(
				target_state_dict['_action_masks'].shape[0] // source_state_dict['_task_emb.weight'].shape[0], 1)
		
	if '_task_emb.weight' in source_state_dict and not '_task_emb.weight' in target_state_dict:
		# delete task embedding from source state dict
		source_state_dict.pop('_task_emb.weight', None)

	return source_state_dict


def print_mismatched_tensors(target_state_dict, source_state_dict):
	target_keys = set(target_state_dict.keys())
	source_keys = set(source_state_dict.keys())

	# Keys in source but not in target
	for key in source_keys - target_keys:
		print(f"[Extra in source] {key}: shape={tuple(source_state_dict[key].shape)}")

	# Keys in target but not in source
	for key in target_keys - source_keys:
		print(f"[Missing in source] {key}: expected shape={tuple(target_state_dict[key].shape)}")

	# Keys present in both but with shape mismatch
	for key in target_keys & source_keys:
		try:
			t_shape = tuple(target_state_dict[key].shape)
		except AttributeError as e:
			print(f"[Error accessing shape in target_state_dict] {key}: {e}")
			continue
		try:
			s_shape = tuple(source_state_dict[key].shape)
		except AttributeError as e:
			print(f"[Error accessing shape in source_state_dict] {key}: {e}")
			continue
		if t_shape != s_shape:
			print(f"[Shape mismatch] {key}: target={t_shape}, source={s_shape}")
