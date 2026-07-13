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
                "local_refinement": getattr(args, "local_refinement", False),
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

    def train(self, train_data, device, args, mode, round_idx = None):

        # mode 0 :  training with mask
        # mode 1 : training with mask
        # mode 2 : training with mask, calculate mask
        # mode 3 : training with mask, calculate mask
        model = self.model
        model.to(device)
        model.train()

        # auto-disable weight decay when L1 regularization is active
        if getattr(args, "reg_mode", "") == "l1" and getattr(args, "reg_weight", 0.0) > 0:
            logging.warning("L1 reg active: forcing weight decay to 0")
            args.wd = 0.0

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        assert trainable_params, "no trainable parameters found for optimizer"
        # verify weight decay is 0 when L1 is active
        if getattr(args, "reg_mode", "") == "l1" and getattr(args, "reg_weight", 0.0) > 0:
            assert args.wd == 0.0, "weight decay must be 0 when L1 reg is active"
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

        # ── 预训练 CDF 剪枝 (替代 ClientManager.local_refinement，有数据可用梯度指标) ──
        adjust_type = getattr(args, "adjustment_type", None)
        if getattr(args, "local_refinement", False) and mode in [0, 3] and adjust_type:
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

                x, labels = next(iter(train_data))
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                loss = criterion(model(x), labels)
                loss.backward()
                model.general_cdf_prune(p=args.p, adjustment_type=adjust_type)
                model.apply_mask()

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

            # local refinement skips mode 2 prune/grow (already done on mask receipt)
            if getattr(args, "local_refinement", False):
                pass
            elif model.has_gated_convs():
                model.prune_by_gate_cdf(p=args.p)
            elif adjust_type == "mag_cdf":
                model.general_cdf_prune(p=args.p, adjustment_type="mag_cdf")
            elif adjust_type == "channel_l1_cdf":
                model.channel_l1_cdf_prune(p=args.p)
            else:
                # original FedDST prune+grow maintains density
                model.adjust_mask_dict(gradients, t=round_idx, T_end=args.T_end, alpha=args.adjust_alpha)
                model.apply_mask()

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

        # ── weight_archive: record trained positions after local round ──
        if getattr(args, "weight_archive", False) and self.model.weight_archive is not None:
            self.model.update_weight_archive(self.model.mask_dict)
        # ───────────────────────────────────────────────────────────────

        return model.mask_dict

    def test(self, test_data, device, args, **kwargs):
        model = self.model

        model.to(device)
        model.eval()

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

                metrics['Accuracy'] += correct.item()
                metrics['Loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        metrics['Accuracy'] /= metrics['test_total']
        metrics['Loss'] /= metrics['test_total']
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
