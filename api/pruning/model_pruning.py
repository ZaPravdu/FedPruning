import torch
from typing import Dict, List
from torch import nn
from api.pruning.init_scheme import generate_layer_density_dict, pruning, sparse_update_step, sparse_pruning_step, sparse_growing_step, compute_cdf_metric, cdf_prune_by_metric
from api.model.cv.resnet import iter_gated_convs, iter_vd_convs
import warnings
import logging
import re

class SparseModel(nn.Module):
    def __init__(self, model,
                 target_density:float=0.5,
                 strategy:str="ERK_magnitude",
                 mask_dict: dict = {},
                 ignore_layers:list[int, str, type]=[".*bias.*", ".*\.gate$", ".*\.log_sigma2$", nn.BatchNorm2d, ".*bn.*", nn.LayerNorm, ".*ln.*"],
                 device = None,
                 init_density = None,
                 ):
        super(SparseModel, self).__init__()
        # strategy is a str that [sparsity_distribution]_[pruning_strategy]
        # e.g. uniform_magnitude

        self.model = model
        self.mask_dict = mask_dict
        self.strategy = strategy
        self.target_density = target_density      # static ultimate target
        self.ignore_layers = ignore_layers
        self.device = device


        # layer_set includes all layer names
        # layer_shape_dict includes the shape of every layer
        # num_overall_elements is the number of parameters in the whole model
        self.layer_set, self.layer_shape_dict, self.num_overall_elements = self._stat_layer_info()

        # mask_dict only includes the mask of layer that should be pruned(a.k.a sparse layer)
        # sparse_layer_set the name of the sparse layer
        # layer_density_dict includes the layer-wise densities for sparse layer (not include ignored layers)

        if self.mask_dict:
            self.sparse_layer_set = set(self.mask_dict.keys())
            logging.debug("########### call mask dict here #########")
        else:
            self.sparse_layer_set = self._determine_sparse_layers()

        layer_strat, pruning_strat = self.strategy.split("_")

        # Static floor: ERK from target_density (NEVER changes)
        self.layer_density_dict = generate_layer_density_dict(
            self.layer_shape_dict, self.num_overall_elements,
            self.sparse_layer_set, self.target_density, layer_strat,
        )

        # Dynamic density: for pruning / scheduling
        self.current_density = init_density if init_density is not None else self.target_density
        self.current_layer_density_dict = generate_layer_density_dict(
            self.layer_shape_dict, self.num_overall_elements,
            self.sparse_layer_set, self.current_density, layer_strat,
        )

        # Initialize masks from current_density (if not from checkpoint)
        if not self.mask_dict:
            self.mask_dict = pruning(self.model, self.current_layer_density_dict, pruning_strat)

        logging.info(f"Sparse layers (floor): {self.layer_density_dict}")

    def to(self, device, *args, **kwargs):
        self.device = device
        self.model.to(device, *args, **kwargs)
        for name in self.mask_dict:
            self.mask_dict[name] = self.mask_dict[name].to(device, *args, **kwargs)

    def _determine_sparse_layers(self):
        sparse_layer_set = self.layer_set.copy()
        ignore_partial_names = []
        ignore_layer_idx = []
        ignore_nn_types = []
        module_length = 0
        for _ in self.model.named_modules():
            module_length += 1

        for item in self.ignore_layers:
            if isinstance(item, str):
                ignore_partial_names.append(item)
            elif isinstance(item, int):
                ignore_layer_idx.append(item)
            elif type(item) is type:
                ignore_nn_types.append(item)
            else:
                warnings.warn(f"{type(item)} is not included in int, str and class. Therefore it will be ignored")

        def _remove_by_name(layer_set, partial_name):
            ###### remove partial names (can use prefix)########
            for layer_name in list(layer_set):
                if re.match(partial_name, layer_name) is not None:
                    layer_set.remove(layer_name)
                # elif partial_name + ".weight" in layer_name:
                #     sparse_layer_set.remove(layer_name)
            return layer_set

        for partial_name in ignore_partial_names:
            sparse_layer_set = _remove_by_name(sparse_layer_set, partial_name,)

        for e, (name, module) in enumerate(self.model.named_modules()):
            # if name == "":
            #     continue

            # if e in ignore_layer_idx:
            #     sparse_layer_set.remove(name)
            #     continue
            for t in ignore_nn_types:
                if isinstance(module, t):
                    sparse_layer_set = _remove_by_name(sparse_layer_set, name)
                    break
        
        # total_length = len(sparse_layer_set)
        # for i in range(len(ignore_layer_idx)):
        #     if ignore_layer_idx[i] < 0:
        #         ignore_layer_idx[i] += total_length
        # # must sorted
        # ignore_layer_idx.sort(reverse=True)
        # sparse_layer_set = list(sparse_layer_set)
        # for idx in ignore_layer_idx:
        #     sparse_layer_set.pop(idx)
        # sparse_layer_set = set(sparse_layer_set)
        return sparse_layer_set


    def _stat_layer_info(self):
        layer_set = set()
        layer_shape_dict = {}
        num_overall_elements = 0
        for name, weight in self.model.named_parameters():
            layer_set.add(name)
            layer_shape_dict[name] = weight.shape
            num_overall_elements += weight.numel()
        return layer_set, layer_shape_dict, num_overall_elements

    def _stat_density_info(self):
        layer_density_dict = 0
        for name, weight in self.model.named_parameters():
            if name in self.mask_dict:
                remains = self.mask_dict[name].sum().item()
                overall = self.mask_dict[name].numel()
                layer_density_dict[name] = remains / overall

        return  layer_density_dict


    def _init_prune(self, density=None):
        layer_density_strategy, pruning_strategy = self.strategy.split("_")
        if density is None:
            density = self.target_density
        layer_density_dict = generate_layer_density_dict(
            self.layer_shape_dict, self.num_overall_elements,
            self.sparse_layer_set, density, layer_density_strategy,
        )
        model_mask = pruning(self.model, layer_density_dict, pruning_strategy)
        return layer_density_dict, model_mask

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    def named_parameters(self, **kwargs):
        return self.model.named_parameters(**kwargs)

    def has_gated_convs(self):
        return any(True for _ in iter_gated_convs(self.model))

    def has_vd_convs(self):
        return any(True for _ in iter_vd_convs(self.model))

    def has_vd_original_mode(self):
        """True if any VDConv2d is in 'original' (per-weight) mode."""
        for _, mod in iter_vd_convs(self.model):
            if mod.mode == "original":
                return True
        return False

    def compute_vd_regularization(self):
        """Sum of per-channel KL losses across all VDConv2d modules."""
        total_kl = []
        for _, vd_conv in iter_vd_convs(self.model):
            kl = vd_conv.kl_loss()
            # total_kl = kl if total_kl is None else total_kl + kl
            total_kl.append(kl)
        return torch.stack(total_kl).mean()

    def compute_ns_regularization(self):
        """Network Slimming: L1 norm of gamma (weight) in all normalization layers.

        Collects .weight from BatchNorm/LayerNorm/GroupNorm/InstanceNorm modules
        and computes sum(|gamma|) as a sparsity-inducing regularizer.
        """
        device = next(self.model.parameters()).device
        gamma_l1 = torch.tensor(0.0, device=device)
        count = 0
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                      nn.LayerNorm, nn.GroupNorm,
                      nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
        for module in self.model.modules():
            if isinstance(module, norm_types):
                gamma_l1 = gamma_l1 + module.weight.abs().sum()
                count += 1
        assert count > 0, "No normalization layers found for NS regularization"
        return gamma_l1

    @staticmethod
    def _nard_group_layout(module):
        """Determine NARD group layout for a norm module.

        Returns (num_groups, total_channels, gamma_per_group).

        * BN/IN:   per-channel (num_groups = C)
        * GN (custom _GroupNorm): gamma already per-group, shape (G,)
        * GN (nn.GroupNorm):      gamma per-channel,    summed per group
        * LN:                      single group
        """
        gamma = module.weight
        if isinstance(module, nn.GroupNorm):
            try:
                from api.model.cv.group_normalization import _GroupNorm
            except ImportError:
                _GroupNorm = type(None)
            if isinstance(module, _GroupNorm):
                num_groups = module.num_groups
                total_channels = module.num_features * num_groups
                gamma_per_group = gamma
            else:
                num_groups = module.num_groups
                total_channels = module.num_channels
                chpg = total_channels // num_groups
                gamma_per_group = gamma.view(num_groups, chpg).sum(dim=1)
        elif isinstance(module, nn.LayerNorm):
            num_groups = 1
            total_channels = gamma.numel()
            gamma_per_group = gamma.sum(dim=0, keepdim=True)
        else:  # BN / IN
            num_groups = gamma.numel()
            total_channels = num_groups
            gamma_per_group = gamma
        return num_groups, total_channels, gamma_per_group

    def compute_nard_regularization(self):
        """NARD: sum_i 1/gamma_i^2 * ||w_i||^2 + sum_i log(gamma_i^2).

        DFS-walks modules(), pairs each norm with the preceding weight module
        (Conv/Linear), then iterates groups (BN->channels, GN->groups).
        """
        device = next(self.model.parameters()).device
        total = torch.tensor(0.0, device=device)
        norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                      nn.LayerNorm, nn.GroupNorm,
                      nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
        weight_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
        last_weight = None
        pair_count = 0

        for module in self.model.modules():
            if isinstance(module, weight_types):
                last_weight = module
            elif isinstance(module, norm_types):
                assert last_weight is not None, \
                    f"NARD: {type(module).__name__} has no preceding weight layer"
                num_groups, total_ch, gamma_per_group = self._nard_group_layout(module)
                w = last_weight.weight
                assert w.size(0) == total_ch, (
                    f"NARD dim mismatch: {type(module).__name__} has {total_ch} "
                    f"features but paired weight has {w.size(0)} output channels"
                )
                ch_per_group = total_ch // num_groups
                w_flat = w.view(total_ch, -1)
                w_per_ch = w_flat.pow(2).sum(dim=1)
                w_per_grp = w_per_ch.view(num_groups, ch_per_group).sum(dim=1)
                gamma_sq = gamma_per_group.pow(2)
                total = total + (w_per_grp / gamma_sq).sum() + gamma_sq.log().sum()
                pair_count += 1

        assert pair_count > 0, "NARD: no (weight, norm) pairs found"
        return total

    def compute_l1_regularization(self):
        """L1 norm of weight matrices for pruned layers (no mask applied)."""
        l1 = torch.tensor(0.0, device=next(self.model.parameters()).device)
        count = 0
        for name, param in self.model.named_parameters():
            if name in self.mask_dict:
                l1 = l1 + param.abs().sum()
                count += 1
        assert count > 0, "No masked layers found for L1 regularization"
        return l1

    def set_gate_trainable(self, trainable: bool):
        for _, module in iter_gated_convs(self.model):
            module.set_gate_trainable(trainable)

    def compute_regularization(self, reg_type=None, eps=1e-6):
        total_reg = None
        for _, module in iter_gated_convs(self.model):
            gate = module.gate
            weight = module.conv.weight
            if reg_type == "gate_l1":
                reg = gate.abs().sum()
            elif reg_type == "weight_l1_over_gate":
                w_per_ch = weight.view(weight.size(0), -1).abs().sum(dim=1)
                reg = (w_per_ch / (gate + eps)).sum()
            elif reg_type == "weight_l2_over_gate":
                w_per_ch = weight.view(weight.size(0), -1).pow(2).sum(dim=1)
                reg = (w_per_ch / (gate + eps)).sum()
            else:
                continue
            total_reg = reg if total_reg is None else total_reg + reg
        assert total_reg is not None, f"No gated convs found for reg_type={reg_type}"
        return total_reg

    @torch.no_grad()
    def compute_gate_guided_density(self):
        # collect gated conv weight names for gate-aware counting
        gated_weight_names = set()
        gate_masks = {}
        for name, module in iter_gated_convs(self.model):
            wn = f"{name}.conv.weight"
            gated_weight_names.add(wn)
            gate_masks[wn] = (module.gate != 0).bool()

        num_remain_elements = 0
        total_elements = 0
        for name, weight in self.model.named_parameters():
            total_elements += weight.numel()
            if name in gated_weight_names:
                # gate-aware: count elements where BOTH pruning mask AND gate are 1
                if name in self.mask_dict:
                    effective = self.mask_dict[name].clone()
                    effective[~gate_masks[name]] = 0.0
                    num_remain_elements += (effective != 0.0).sum().item()
                else:
                    num_remain_elements += weight[gate_masks[name]].numel()
            else:
                # use mask_dict for masked params (weight != 0 can be wrong due to momentum
                # filling in zeroed weights after optimizer.step()); fall back to weight
                # values for non-masked params (bias, BN, etc.)
                if name in self.mask_dict:
                    num_remain_elements += (self.mask_dict[name] != 0.0).sum().item()
                else:
                    num_remain_elements += torch.sum(weight != 0.).item()

        if total_elements == 0:
            return 0.0
        return num_remain_elements / total_elements

    @torch.no_grad()
    def compute_vd_density(self):
        """Effective density accounting for both mask_dict AND VD eval pruning.

        For weights under VDConv2d layers, an element is "alive" only when
        both mask_dict[name]=1 AND log_alpha < thresh (i.e. VD eval won't prune it).
        Non-VD masked params are counted from mask_dict as usual.
        Non-masked params (bias, BN, etc.) are always counted.
        """
        # collect VD pruning masks  (1 = alive i.e. log_alpha < thresh)
        vd_masks = {}
        for name, mod in iter_vd_convs(self.model):
            W = mod.conv.weight
            log_alpha = mod._clip(mod.log_sigma2 - torch.log(W * W + 1e-8))
            vd_masks[f"{name}.conv.weight"] = (log_alpha < mod.thresh).float()

        num_remain = 0
        total = 0
        for name, weight in self.model.named_parameters():
            total += weight.numel()
            if name in vd_masks:
                # combine mask_dict with VD pruning mask
                vd_mask = vd_masks[name].to(self.mask_dict[name].device)
                if name in self.mask_dict:
                    effective = self.mask_dict[name] * vd_mask
                else:
                    effective = vd_mask
                num_remain += (effective != 0).sum().item()
            elif name in self.mask_dict:
                num_remain += (self.mask_dict[name] != 0).sum().item()
            else:
                # unprunable params (bias, BN, log_sigma2 itself) — always count
                num_remain += weight.numel()

        return num_remain / total if total > 0 else 0.0

    @torch.no_grad()
    def reopen_gated_channels(self):
        for name, module in iter_gated_convs(self.model):
            module.reopen_gates()

    @torch.no_grad()
    def prune_by_gate_cdf(self, p=0.85):
        assert 0.0 < p <= 1.0
        prune_stats = {}
        for name, module in iter_gated_convs(self.model):
            keep_mask = module.prune_by_cdf(p)
            prune_stats[name] = {
                "kept": int(keep_mask.sum().item()),
                "total": int(keep_mask.numel()),
            }

            weight_name = f"{name}.conv.weight"
            if weight_name in self.mask_dict:
                self.mask_dict[weight_name][~keep_mask] = 0.0

            bias_name = f"{name}.conv.bias"
            if bias_name in self.mask_dict:
                self.mask_dict[bias_name][~keep_mask] = 0.0

        self.apply_mask()
        return prune_stats

    @torch.no_grad()
    def general_cdf_prune(self, p=0.85, adjustment_type="mag_cdf"):
        """Generic element-wise CDF pruning with configurable metric.

        For each masked layer: computes an importance metric matrix
        via compute_cdf_metric, then applies CDF pruning via
        cdf_prune_by_metric to retain the active elements covering
        fraction p of total metric sum.

        If layer_density_dict (from target_density) is set, each layer keeps at least
        target_density × num_elements elements (CDF lock).

        Metric type (e.g. "mag", "magnitude") is controlled by
        adjustment_type — extend compute_cdf_metric() to add new ones.

        Args:
            p: fraction of total metric sum to retain (0 < p <= 1)
            adjustment_type: metric type forwarded to compute_cdf_metric
        Returns:
            prune_stats dict per layer
        """
        assert 0.0 < p <= 1.0, f"p must be in (0,1], got {p}"
        prune_stats = {}
        num_remain_elements = 0
        num_total_elements = 0
        for name, weight in self.model.named_parameters():
            if name not in self.mask_dict:
                continue
            mask = self.mask_dict[name]
            active_num = int((mask.view(-1) != 0).sum().item())
            num_total_elements += active_num
            if active_num == 0:
                continue

            # floor lock: at least target_density × num_elements per layer
            min_keep = None
            if self.layer_density_dict and name in self.layer_density_dict:
                min_keep = int(weight.numel() * self.layer_density_dict[name])

            metric = compute_cdf_metric(weight, mask, adjustment_type)
            new_mask = cdf_prune_by_metric(metric, weight, mask, p, min_keep=min_keep)

            keep_count = int((new_mask.view(-1) != 0).sum().item())
            num_remain_elements += keep_count
            self.mask_dict[name] = new_mask
            prune_stats[name] = {"kept": keep_count, "total": active_num}

        self.apply_mask()
        logging.debug(f"{adjustment_type}_cdf_prune: kept {num_remain_elements}/{num_total_elements} "
                      f"({num_remain_elements/num_total_elements:.4f})")
        return prune_stats

    @torch.no_grad()
    def channel_l1_cdf_prune(self, p=0.85):
        """Structured channel-level pruning via L1-norm CDF.

        For each conv weight (dim >= 4) in mask_dict: compute L1 norm of
        each output channel, sort descending, keep channels covering
        fraction p of total L1 sum, prune the rest (zero entire channel
        in mask_dict).  Structured-channel analog of magnitude_cdf_prune
        for plain (non-gated) ResNet."""
        assert 0.0 < p <= 1.0
        prune_stats = {}
        for name, weight in self.model.named_parameters():
            if name not in self.mask_dict:
                continue
            if weight.dim() < 4:
                # skip non-conv layers (linear, bias — no channel dim)
                continue

            mask = self.mask_dict[name]
            # per-channel L1 norm of masked weight
            w = weight.data * mask
            per_ch_l1 = w.view(w.size(0), -1).abs().sum(dim=1)  # [C_out]

            active_ch = (per_ch_l1 != 0)
            active_num = active_ch.sum().item()
            if active_num == 0:
                continue

            # CDF over active channels only
            active_l1 = per_ch_l1[active_ch]
            sorted_vals, idx = torch.sort(active_l1, descending=True)
            total_l1 = sorted_vals.sum()
            if total_l1 == 0:
                continue
            cumsum = torch.cumsum(sorted_vals, dim=0)
            keep_count = int((cumsum < p * total_l1).sum().item()) + 1
            keep_count = max(1, min(keep_count, active_num))

            # build channel-level keep mask and apply
            active_positions = torch.where(active_ch)[0]
            keep_ch = torch.zeros(w.size(0), dtype=torch.bool, device=w.device)
            keep_ch[active_positions[idx[:keep_count]]] = True

            self.mask_dict[name][~keep_ch] = 0.0

            prune_stats[name] = {
                "kept_channels": keep_count,
                "total_channels": w.size(0),
            }

        self.apply_mask()
        return prune_stats

    @torch.no_grad()
    def apply_mask(self,):
        for name, weight in self.model.named_parameters():
            if name in self.mask_dict:
                try:
                    weight.data = weight.data * self.mask_dict[name]
                except RuntimeError:
                    raise RuntimeError(f"the device for weight is {weight.device} and mask_dict is on {self.mask_dict[name].device}") 
                
    @torch.no_grad()
    def apply_mask_gradients(self):
        """
        Applies boolean mask to modules's gradients
        """
        for name, weight in self.module.named_parameters():
            if name in self.mask_dict:
                weight.grad = weight.grad * self.mask_dict[name]

    def forward(self, x, *args, **kargs):
        # mask weight
        self.apply_mask()
        y = self.model(x, *args, **kargs)
        return y

    def stat_actual_density(self):
        num_remain_elements = 0
        actual_layer_wise_density = {}
        for name, weight in self.model.named_parameters():
            if name in self.mask_dict:
                layer_remain_elements = (self.mask_dict[name] != 0.).sum().item()
            else:
                layer_remain_elements = torch.sum(weight != 0.).item()
            num_remain_elements += layer_remain_elements
            actual_layer_wise_density[name] = layer_remain_elements / weight.numel()

        actual_density = num_remain_elements/ self.num_overall_elements

        return actual_density, actual_layer_wise_density
    
    def adjust_mask_dict(self, gradients, t, T_end, alpha):
        self.mask_dict = sparse_update_step(self.model, gradients, self.mask_dict, t, T_end, alpha)

    def prune_and_grow_fedsgc(self, weights, masks, gradient_dict, local_direction_map, t, alpha, T_end, lambda_k, beta_k, global_direction_map):
        if global_direction_map is None:
            logging.warning("global_direction_map is None. Initializing it as an empty dictionary.")
            global_direction_map = {}

        # Assign global and local direction maps for easier reference.
        d_t = global_direction_map
        delta = local_direction_map

        # Iterate through all the keys in the weights dictionary.
        for key in weights:
            # Validate if the key exists in masks, global_direction_map (d_t), and local_direction_map (delta).
            if key not in masks or key not in d_t or key not in delta:
                logging.warning(f"Skipping invalid or missing key: {key}")
                continue

            # Retrieve weight, mask, global direction, and local direction for the current key.
            weight = weights[key]
            mask = masks[key]
            global_direction = d_t[key]
            local_direction = delta[key]

            # Identify active (mask == 1) and inactive (mask == 0) weight indices.
            active_indices = (mask == 1).nonzero(as_tuple=True)[0]
            inactive_indices = (mask == 0).nonzero(as_tuple=True)[0]
            active_num = len(active_indices)

            # Determine the number of weights to prune based on the current round and alpha parameter.
            k = int(((1 - t / T_end) ** alpha) * active_num)

            # Pruning strategy 1: Prune weights where global_direction aligns oppositely with local_direction.
            valid_prune_indices_1 = active_indices[global_direction[active_indices] == -local_direction[active_indices]]
            if len(valid_prune_indices_1) > 0:
                sorted_active_by_weight_1 = valid_prune_indices_1[
                    torch.argsort(torch.abs(weight[valid_prune_indices_1]))
                ]
                num_to_prune_1 = min(int(lambda_k * k), len(sorted_active_by_weight_1))
                mask[sorted_active_by_weight_1[:num_to_prune_1]] = 0
                logging.info(f"Pruned {num_to_prune_1} weights from {key} ([d_t]_i = -[Δ]_i).")

            # Pruning strategy 2: Prune weights where global_direction does not align oppositely with local_direction.
            valid_prune_indices_2 = active_indices[global_direction[active_indices] != -local_direction[active_indices]]
            if len(valid_prune_indices_2) > 0:
                sorted_active_by_weight_2 = valid_prune_indices_2[
                    torch.argsort(torch.abs(weight[valid_prune_indices_2]))
                ]
                num_to_prune_2 = min(int((1 - lambda_k) * k), len(sorted_active_by_weight_2))
                mask[sorted_active_by_weight_2[:num_to_prune_2]] = 0
                logging.info(f"Pruned {num_to_prune_2} weights from {key} ([d_t]_i ≠ -[Δ]_i).")
            
            # what is trainer ? 
            # Retrieve the gradient for the current key. Skip growing if the gradient is not found.
            # gradient = self.trainer.get_gradient(key)
            # if gradient is None:
            #     logging.error(f"Gradient for {key} not found. Skipping growth for this key.")
            #     continue

            # add gradients dict 
            gradient = gradient_dict[key]

            # Growing strategy 1: Grow weights where global_direction aligns with local_direction.
            valid_grow_indices_1 = inactive_indices[
                global_direction[inactive_indices] == local_direction[inactive_indices]]
            if len(valid_grow_indices_1) > 0:
                sorted_inactive_by_grad_1 = valid_grow_indices_1[
                    torch.argsort(torch.abs(gradient[valid_grow_indices_1]), descending=True)
                ]
                num_to_grow_1 = min(int(beta_k * k), len(sorted_inactive_by_grad_1))
                mask[sorted_inactive_by_grad_1[:num_to_grow_1]] = 1
                logging.info(f"Grew {num_to_grow_1} weights for {key} ([d_t]_i = [Δ]_i).")

            # Growing strategy 2: Grow weights where global_direction does not align with local_direction.
            valid_grow_indices_2 = inactive_indices[
                global_direction[inactive_indices] != local_direction[inactive_indices]]
            if len(valid_grow_indices_2) > 0:
                sorted_inactive_by_grad_2 = valid_grow_indices_2[
                    torch.argsort(torch.abs(gradient[valid_grow_indices_2]), descending=True)
                ]
                num_to_grow_2 = min(int((1 - beta_k) * k), len(sorted_inactive_by_grad_2))
                mask[sorted_inactive_by_grad_2[:num_to_grow_2]] = 1
                logging.info(f"Grew {num_to_grow_2} weights for {key} ([d_t]_i ≠ [Δ]_i).")

    def prune_mask_dict(self, t, T_end, alpha):
        self.mask_dict = sparse_pruning_step(self.model, self.mask_dict, t, T_end, alpha)
    def grow_mask_dict(self, gradients):
        self.mask_dict = sparse_growing_step(self.model, gradients, self.mask_dict, self.current_layer_density_dict)
if __name__ == "__main__":
    from torchvision.models import resnet18
    model = resnet18()
    sparse_model = SparseModel(model, target_density=0.5, )
    #sparse_model.apply_mask()
    sparse_layer_set = sparse_model.sparse_layer_set
    print("#########ignored layers##########")
    print(sparse_model.layer_set - sparse_layer_set)
    print("###############sparse layers ##########")
    print(sparse_layer_set)
    print("#############density distribution#############")
    print(sparse_model.layer_density_dict)

    ## training
    sparse_model.to("cuda")
    optim = optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, sparse_model.parameters()), lr=0.1)
    for i in range(10):
        sparse_model.zero_grad()
        x = torch.randn([32, 3, 32, 32]).cuda()
        y = sparse_model(x)
        loss = torch.sum(y * torch.randn_like(y))
        loss.backward()
        optimizer.step()

    print("##############recheck the parameter in the training################")
    _ = sparse_model(x)
    sparse_model.zero_grad()
    actual_density, actual_layer_wise_density = sparse_model.stat_actual_density()
    print("######### actual overall density ###########")
    print(actual_density)
    print("######### actual layer wise density ###########")
    print(actual_layer_wise_density)
