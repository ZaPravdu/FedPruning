import json
import logging
import os

import torch
from torch import nn
from ...pruning.init_scheme import f_decay

try:
    from core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedPruning.core.trainer.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):

    def __init__(self, model, args=None):
        super().__init__(model)
        self.diagnostics = {
            "rounds": [],
            "config": {
                "init_density": getattr(args, "init_density", None),
                "target_density": getattr(args, "target_density", None),
                "p": getattr(args, "p", None),
                "reg_mode": getattr(args, "reg_mode", None),
                "reg_weight": getattr(args, "reg_weight", None),
                "reg_adjust_only": getattr(args, "reg_adjust_only", False),
                "cdf_pos": getattr(args, "cdf_pos", "post-train"),
                "adjustment_type": getattr(args, "adjustment_type", None),
                "client_optimizer": getattr(args, "client_optimizer", ""),
                "lr": getattr(args, "lr", None),
            },
            "l1_losses": [],
        }

    def get_model(self):
        return self.model

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def _add_reg(self, args, loss):
        """Apply regularization term to loss based on args.reg_mode."""
        reg_w = getattr(args, "reg_weight", 0.0)
        if reg_w <= 0:
            return loss
        reg_m = getattr(args, "reg_mode", "none")

        if reg_m == "l1":
            loss = loss + reg_w * self.model.compute_l1_regularization()
        elif reg_m == "ns":
            loss = loss + reg_w * self.model.compute_ns_regularization()
        elif reg_m == "nard":
            loss = loss + reg_w * self.model.compute_nard_regularization()
        elif reg_m in ("gate_l1", "weight_l1_over_gate", "weight_l2_over_gate"):
            g = self.model.compute_regularization(reg_type=reg_m, eps=getattr(args, "gate_reg_eps", 1e-6))
            assert g is not None, f"compute_regularization returned None for reg_type={reg_m}"
            loss = loss + reg_w * g
        elif reg_m in ("channel", "original"):
            loss = loss + reg_w * self.model.compute_vd_regularization()
        return loss

    # ── CDF pruning (moved from SparseModel.general_cdf_prune) ──

    def compute_gradients(self):
        """Compute gradients for one batch from self._train_data.

        Returns dict of {name: param.grad.clone()} for all trainable params.
        """
        assert self._train_data is not None, "self._train_data not set"
        model = self.model
        device = next(model.parameters()).device
        model.zero_grad()
        x, labels = next(iter(self._train_data))
        x, labels = x.to(device), labels.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(model(x), labels)
        loss.backward()
        grads = {name: param.grad.clone()
                 for name, param in model.named_parameters()
                 if param.grad is not None}
        model.zero_grad()
        return grads

    def compute_cdf_metric(self, adjustment_type):
        """Compute CDF importance metric per masked layer.

        Returns {name: metric_tensor} for layers in model.mask_dict.
        Raises ValueError on unknown adjustment_type.
        """
        model = self.model

        if adjustment_type == "mag_cdf":
            return {name: param.data.abs()
                    for name, param in model.named_parameters()
                    if name in model.mask_dict}

        elif adjustment_type == "mag_grad_mag":
            grads = self.compute_gradients()
            metrics = {}
            for name, param in model.named_parameters():
                if name not in model.mask_dict:
                    continue
                if name not in grads:
                    raise RuntimeError(
                        f"mag_grad_mag: no gradient for {name}; "
                        "ensure the parameter requires_grad"
                    )
                mag_src = param.data
                metrics[name] = grads[name].abs() * mag_src.abs()
            return metrics

        else:
            raise ValueError(f"Unknown CDF metric adjustment_type: {adjustment_type}")

    def cdf_prune(self, p, adjustment_type):
        """Orchestrate CDF pruning across all masked layers.

        1. Calls compute_cdf_metric to get importance metrics
        2. Applies cdf_prune_by_metric per layer in mask_dict
        3. Calls model.apply_mask() to materialize the new mask

        Replaces SparseModel.general_cdf_prune — now hosted on MyModelTrainer
        so gradient-dependent metrics (mag_grad_mag) can compute needed data.
        """
        assert adjustment_type is not None, "adjustment_type must not be None"
        from ...pruning.init_scheme import cdf_prune_by_metric

        metrics = self.compute_cdf_metric(adjustment_type)
        model = self.model

        for name, mask in model.mask_dict.items():
            if name not in metrics:
                continue
            min_keep = None
            if getattr(model, 'layer_density_dict', None) and name in model.layer_density_dict:
                min_keep = int(mask.numel() * model.layer_density_dict[name])
            model.mask_dict[name] = cdf_prune_by_metric(metrics[name], mask, p, min_keep=min_keep)

        model.apply_mask()

    # ─────────────────────────────────────────────────────────────────────────

    def train(self, train_data, device, args, mode, round_idx = None):

        # mode 0 :  training with mask
        # mode 1 : training with mask
        # mode 2 : training with mask, calculate mask
        # mode 3 : training with mask, calculate mask
        model = self.model
        model.to(device)
        model.train()
        self._train_data = train_data

        # auto-disable weight decay when L1 regularization is active
        if getattr(args, "reg_mode", "") == "l1" and getattr(args, "reg_weight", 0.0) > 0:
            logging.warning("L1 reg active: forcing weight decay to 0")
            args.wd = 0.0

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        assert trainable_params, "no trainable parameters found for optimizer"
        # verify weight decay is 0 when L1 is active

        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=args.lr)
        else:
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        l1_losses_epoch = []  # diagnostic: track L1 loss per step

        if mode in [2, 3]:
            local_epochs = args.adjustment_epochs if args.adjustment_epochs is not None else args.epochs
        else:
            local_epochs = args.epochs

        if mode in [2, 3]:
            A_epochs = local_epochs // 2 if args.A_epochs is None else args.A_epochs
            first_epochs = min(local_epochs, A_epochs)
        else:
            first_epochs = local_epochs

        if mode in [2, 3] and getattr(args, "model", "") == "gated_resnet18":
            if getattr(args, "reopen_gate_on_adjust", 1) and model.has_gated_convs():
                model.reopen_gated_channels()

        # ── 预训练 CDF 剪枝 (cdf_pos=pre-train: 收到 mask 后先剪再训) ──
        adjust_type = getattr(args, "adjustment_type", None)
        if getattr(args, "cdf_pos", "post-train") == "pre-train" and mode in [0, 3] and adjust_type:
            # ── diagnostic: capture density + weight stats before CDF ──
            pre_density, pre_layer = model.stat_actual_density()
            # active weight statistics across all pruned layers
            active_weights = []
            for name, w in model.named_parameters():
                if name in model.mask_dict:
                    m = model.mask_dict[name]
                    active = w.data[m != 0].flatten()
                    if active.numel() > 0:
                        active_weights.append(active)
            wstats = {}
            if active_weights:
                all_a = torch.cat(active_weights)
                nz = all_a.numel()
                if nz > 0:
                    sorted_a, _ = torch.sort(all_a.abs())
                    wstats = {
                        "mean": all_a.mean().item(),
                        "std": all_a.std().item(),
                        "p10": sorted_a[max(1, int(0.10 * nz)) - 1].item(),
                        "p50": sorted_a[max(1, int(0.50 * nz)) - 1].item(),
                        "p90": sorted_a[max(1, int(0.90 * nz)) - 1].item(),
                        "min": sorted_a[0].item(),
                        "max": sorted_a[-1].item(),
                        "n_nonzero": nz,
                    }
            # ───────────────────────────────────────────────────────────

            # CDF-based methods prune here; "mag" (original FedDST) never prunes in mode 0/3
            if adjust_type in ("mag_cdf", "mag_grad_mag"):
                self.cdf_prune(p=args.p, adjustment_type=adjust_type)
            # "mag": no pre-training pruning — original FedDST adjusts in mode 2 only

            # ── diagnostic: capture density after CDF ──
            post_density, post_layer = model.stat_actual_density()
            keep_ratio = post_density / max(pre_density, 1e-8)
            self.diagnostics["rounds"].append({
                "round": round_idx,
                "mode": mode,
                "density_before_cdf": pre_density,
                "density_after_cdf": post_density,
                "cdf_keep_ratio": keep_ratio,
                "weight_stats": wstats,
            })
            # ─────────────────────────────────────────────
        # ───────────────────────────────────────────────────────────────────────

        for epoch in range(first_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                if round_idx is not None and 50 <= round_idx <= args.T_end and (not args.reg_adjust_only or mode in (2, 3)):
                    loss_ce = loss.item()
                    loss = self._add_reg(args, loss)
                    l1_losses_epoch.append(loss.item() - loss_ce)
                loss.backward()
                #self.model.apply_mask_gradients()  # apply pruning mask

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        if mode in [2, 3]:
            model.zero_grad()
            if args.growth_data_mode == "random":
                gradients = {name: torch.randn_like(param, device='cpu').clone() for name, param in model.named_parameters() if param.requires_grad}

            elif args.growth_data_mode == "single":
                x, labels = next(iter(train_data))
                x, labels = x[0].unsqueeze(0).repeat(2, 1, 1, 1).to(device), labels[0].unsqueeze(0).repeat(2).to(device)  # Duplicate the sample to create a pseudo-batch
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
                model.zero_grad()
            else:
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    if args.growth_data_mode == "batch":
                        break
                gradients = {name: param.grad.data.cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
                model.zero_grad()

            # mode 2 mask adjustment:
            #   "mag"            → original FedDST prune+grow (density-maintaining)
            #   "mag_cdf" etc.   → no-op: CDF pruning only when server sent a mask (mode 0/3)
            #   pre-train CDF    → no-op (already CDF-pruned on mask receipt)
            if getattr(args, "cdf_pos", "post-train") == "pre-train":
                pass
            elif model.has_gated_convs():
                model.prune_by_gate_cdf(p=args.p)
            elif adjust_type == "mag":
                model.adjust_mask_dict(gradients, t=round_idx, T_end=args.T_end, alpha=args.adjust_alpha)
                model.apply_mask()
            # else: mag_cdf, mag_grad_mag — no mask adjustment in mode 2

        for epoch in range(first_epochs, local_epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                if round_idx is not None and 50 <= round_idx <= args.T_end and not args.reg_adjust_only:
                    loss_ce = loss.item()
                    loss = self._add_reg(args, loss)
                    l1_losses_epoch.append(loss.item() - loss_ce)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

        # ── diagnostic: store L1 loss average for this round ──
        if l1_losses_epoch:
            l1_avg = sum(l1_losses_epoch) / len(l1_losses_epoch)
            # attach to the most recent round entry if it exists, otherwise create a round entry
            round_entry = None
            if self.diagnostics["rounds"] and self.diagnostics["rounds"][-1]["round"] == round_idx:
                round_entry = self.diagnostics["rounds"][-1]
            else:
                round_entry = {"round": round_idx, "mode": mode}
                self.diagnostics["rounds"].append(round_entry)
            round_entry["l1_loss_avg"] = l1_avg
            round_entry["l1_loss_count"] = len(l1_losses_epoch)
        # ──────────────────────────────────────────────────────

        # ── CDF pruning after training (cdf_pos=post-train: 训完再剪) ──
        # Only runs in mode 0/3 (server sent a mask), never in adjustment-only mode 2
        if getattr(args, "cdf_pos", "post-train") == "post-train" and mode in [0, 3]:
            if adjust_type == "mag_grad_mag":
                self.cdf_prune(p=args.p, adjustment_type="mag_grad_mag")
                # ── 验证: mask=0 的位置权重也必须为 0 ──
                _params = dict(model.named_parameters())
                for _name in model.mask_dict:
                    if _name not in _params:
                        continue
                    _mask = model.mask_dict[_name]
                    _zpos = (_mask == 0)
                    if _zpos.any():
                        _w = _params[_name].data
                        assert (_w[_zpos.to(_w.device, non_blocking=True)] == 0).all(), \
                            f"[PRUNE_CHECK] {_name}: mask=0 but weight non-zero after cdf_prune!"
                # ──────────────────────────────────────────────
            elif adjust_type == "mag_cdf":
                self.cdf_prune(p=args.p, adjustment_type=adjust_type)
            # "mag": no post-training CDF pruning — original FedDST adjusts in mode 2 only
        # ──────────────────────────────────────────────────────────────────────

        # ── 保险: 返回前确保权重与 mask 同步 ──
        model.apply_mask()
        # ─────────────────────────────────────

        return model.mask_dict

    def test(self, test_data, device, args, **kwargs):
        model = self.model

        model.to(device)
        model.eval()

        # ── DIAG: 记录 test 前模型的 mask 密度 ──
        pre_test_mask_density = model.compute_gate_guided_density()
        # ──────────────────────────────────────

        metrics = {
            'Accuracy': 0,
            'Loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x, **kwargs)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                # ── DIAG: 第一个 batch forward 后，测实际权重非零比例 ──
                if batch_idx == 0:
                    total_el = 0
                    nonzero_el = 0
                    for name, p in model.named_parameters():
                        total_el += p.numel()
                        nonzero_el += (p != 0).sum().item()
                    actual_density = nonzero_el / max(total_el, 1)
                    post_forward_mask_density = model.compute_gate_guided_density()
                    match = "OK" if abs(actual_density - post_forward_mask_density) < 1e-4 else "MISMATCH"
                    logging.warning(
                        f"[DIAG_TEST] actual_density={actual_density:.6f} "
                        f"mask_density_pre={pre_test_mask_density:.6f} "
                        f"mask_density_post={post_forward_mask_density:.6f} "
                        f"apply_mask={kwargs.get('apply_mask', 'default')} "
                        f"{match}"
                    )
                # ──────────────────────────────────────────────────────

                metrics['Accuracy'] += correct.item()
                metrics['Loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        metrics['Accuracy'] /= metrics['test_total']
        metrics['Loss'] /= metrics['test_total']
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
