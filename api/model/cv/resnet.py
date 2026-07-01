'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

'''
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GatedConv', 'add_gate_to_conv', 'iter_gated_convs',
           'VDConv2d', 'add_vd_to_conv', 'iter_vd_convs',
           'ResNet', 'resnet110']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class GatedConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        assert isinstance(conv, nn.Conv2d)
        self.conv = conv
        self.gate = nn.Parameter(torch.ones(conv.out_channels))
        self.gate_mask = torch.ones(conv.out_channels)

    def _apply(self, fn):
        super()._apply(fn)
        self.gate_mask = fn(self.gate_mask)
        return self

    def forward(self, x):
        out = self.conv(x)
        assert out.size(1) == self.gate.numel()
        effective_gate = self.gate * self.gate_mask
        return out * effective_gate.view(1, -1, 1, 1)

    @torch.no_grad()
    def set_gate_trainable(self, trainable):
        self.gate.requires_grad_(trainable)

    @torch.no_grad()
    def reopen_gates(self):
        self.gate_mask.fill_(1.0)

    @torch.no_grad()
    def prune_by_cdf(self, p=0.85):
        assert 0.0 < p <= 1.0
        active_idx = torch.nonzero(self.gate_mask > 0, as_tuple=False).view(-1)
        if active_idx.numel() == 0:
            keep_idx = torch.argmax(self.gate).view(1)
            new_mask = torch.zeros_like(self.gate_mask)
            new_mask[keep_idx] = 1.0
        else:
            gate_prob = torch.softmax(self.gate[active_idx], dim=0)
            sorted_prob, order = torch.sort(gate_prob, descending=True)
            cumulative_prob = torch.cumsum(sorted_prob, dim=0)
            keep_count = int((cumulative_prob < p).sum().item()) + 1
            keep_count = max(1, min(keep_count, active_idx.numel()))
            keep_idx = active_idx[order[:keep_count]]
            new_mask = torch.zeros_like(self.gate_mask)
            new_mask[keep_idx] = 1.0

        self.gate_mask.copy_(new_mask)
        self.gate.data.mul_(self.gate_mask)
        self.conv.weight.data.mul_(self.gate_mask.view(-1, 1, 1, 1))
        if self.conv.bias is not None:
            self.conv.bias.data.mul_(self.gate_mask)
        return self.gate_mask.bool()


def add_gate_to_conv(module):
    for name, child in module.named_children():
        if isinstance(child, GatedConv):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, GatedConv(child))
            continue
        add_gate_to_conv(child)
    return module


def iter_gated_convs(module):
    for name, child in module.named_modules():
        if isinstance(child, GatedConv):
            yield name, child


class VDConv2d(nn.Module):
    """Conv2d with Variational Dropout (Molchanov et al. 2017).

    Two modes (controlled by ``mode``):
      "channel" (default):
          log_sigma2 shape = [C_out, 1, 1, 1]  (per output channel, broadcasts).
          Eval = standard conv (no pruning — mag_cdf handles that).
          KL   = -mdkl.mean()
          This is our production version integrated with FedDST's mag_cdf pipeline.

      "original":
          log_sigma2 shape = [C_out, C_in, K, K] (per weight, same as W).
          Eval = zero weights where log_alpha >= thresh  (Sparse VD sparsification).
          train_clip option: zero high-log_alpha weights during training too.
          KL   = -mdkl.sum()   (exact Molchanov 2017).
          This follows Conv2DVarDropOutARD from the Theano/Lasagne source exactly.
    """

    def __init__(self, conv: nn.Conv2d, ard_init: float = -10.0,
                 mode: str = "channel",
                 thresh: float = 3.0, train_clip: bool = False):
        super().__init__()
        assert isinstance(conv, nn.Conv2d), \
            f"VDConv2d expects nn.Conv2d, got {type(conv)}"
        assert mode in ("channel", "original"), \
            f"VDConv2d mode must be 'channel' or 'original', got '{mode}'"
        self.conv = conv
        self.mode = mode
        self.thresh = thresh
        self.train_clip = train_clip

        if mode == "original":
            # Per-weight log_sigma2 (same shape as W) — original paper style
            self.log_sigma2 = nn.Parameter(
                torch.full(conv.weight.shape, ard_init)
            )
        else:
            # Per-output-channel log_sigma2, broadcasts over C_in x K x K
            self.log_sigma2 = nn.Parameter(
                torch.full((conv.out_channels, 1, 1, 1), ard_init)
            )

        self._vd_tag = f"conv({conv.in_channels},{conv.out_channels},{conv.kernel_size[0]},mode={mode})"

    # ------------------------------------------------------------------
    # NaN / inf checker (debug helper kept from earlier iterations)
    # ------------------------------------------------------------------
    def _chk(self, t, name, check_neg=False):
        if t is None:
            return
        has_nan = torch.isnan(t).any()
        has_inf = torch.isinf(t).any() if t.is_floating_point() else False
        has_neg = False
        if check_neg and t.is_floating_point():
            has_neg = (t < -1e-6).any().item()
        if has_nan or has_inf or has_neg:
            n_nan = int(torch.isnan(t).sum().item())
            n_inf = int(torch.isinf(t).sum().item()) if t.is_floating_point() else 0
            n_neg = int((t < 0).sum().item()) if check_neg and t.is_floating_point() else 0
            try:
                vmin = torch.nanmin(t).item()
                vmax = torch.nanmax(t).item()
                vavg = torch.nanmean(t).item()
            except:
                vmin = vmax = vavg = float("nan")
            issues = []
            if has_nan: issues.append(f"NaN={n_nan}")
            if has_inf: issues.append(f"Inf={n_inf}")
            if has_neg: issues.append(f"neg={n_neg}")
            raise RuntimeError(
                f"[VD_NAN] {self._vd_tag}.{name}  {' '.join(issues)}/{t.numel()}  "
                f"shape={list(t.shape)}  dtype={t.dtype}  "
                f"min={vmin:.4e}  max={vmax:.4e}  mean={vavg:.4e}  "
                f"log_sigma2=[{self.log_sigma2.data.min().item():.2f}, "
                f"{self.log_sigma2.data.max().item():.2f}]"
            )

    @staticmethod
    def _clip(log_alpha, to=8.0):
        return torch.clamp(log_alpha, min=-to, max=to)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        W = self.conv.weight                  # [C_out, C_in, K, K]
        bias = self.conv.bias
        self._chk(self.log_sigma2, "log_sigma2(param)")
        self._chk(W, "W")

        # log_alpha = clip(log_sigma2 - log(W^2))
        logW = torch.log(W * W + 1e-8)
        self._chk(logW, "log(W²)")
        log_alpha = self._clip(self.log_sigma2 - logW)
        self._chk(log_alpha, "log_alpha")

        if self.training:
            return self._forward_train(x, W, bias, log_alpha)
        else:
            return self._forward_eval(x, W, bias, log_alpha)

    # ------------------------------------------------------------------
    # Train forward
    # ------------------------------------------------------------------
    def _forward_train(self, x, W, bias, log_alpha):
        if self.mode == "original" and self.train_clip:
            # Original: W = T.switch(clip_mask, 0, self.W)  — zero high log_alpha weights
            W_mu = W.clone()
            W_mu[log_alpha >= self.thresh] = 0.0
        else:
            W_mu = W

        # mean
        mu = F.conv2d(x, W_mu, None, self.conv.stride,
                      self.conv.padding, self.conv.dilation,
                      self.conv.groups)
        self._chk(mu, "mu")

        # variance kernel  (original always uses unclipped self.W * self.W)
        kern_var = log_alpha.exp() * W * W
        self._chk(kern_var, "kern_var")
        x_sq = x * x
        self._chk(x_sq, "x_sq")
        sigma_sq = F.conv2d(x_sq, kern_var, None, self.conv.stride,
                            self.conv.padding, self.conv.dilation,
                            self.conv.groups)
        # cuDNN FMA rounding can give tiny negatives even with non-negative inputs
        sigma_sq = torch.clamp(sigma_sq, min=0.0)
        sigma = torch.sqrt(sigma_sq + 1e-8)
        self._chk(sigma, "sigma(before_clamp)", check_neg=True)
        # prevent softmax overflow (exp(x) > float32 at x≈88)
        sigma = torch.clamp(sigma, max=10.0)
        self._chk(sigma, "sigma(after_clamp)")
        noise = torch.randn_like(mu)
        out = mu + sigma * noise
        self._chk(out, "out")

        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)
            self._chk(out, "out+bias")
        return out

    # ------------------------------------------------------------------
    # Eval forward
    # ------------------------------------------------------------------
    def _forward_eval(self, x, W, bias, log_alpha):
        if self.mode == "original":
            # Original: kerns = T.switch(T.ge(log_alpha, thresh), 0, self.W)
            W_eval = W.clone()
            W_eval[log_alpha >= self.thresh] = 0.0
            return F.conv2d(x, W_eval, bias, self.conv.stride,
                            self.conv.padding, self.conv.dilation,
                            self.conv.groups)
        else:
            # channel mode: standard deterministic conv (pruning handled by mag_cdf)
            return F.conv2d(x, W, bias, self.conv.stride,
                            self.conv.padding, self.conv.dilation,
                            self.conv.groups)

    # ------------------------------------------------------------------
    # KL loss
    # ------------------------------------------------------------------
    def kl_loss(self) -> torch.Tensor:
        """Closed-form KL (Molchanov 2017 eq 14-15).

        "channel" mode: -mdkl.mean()
        "original" mode: -mdkl.sum()   — exact match with original source.
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        W = self.conv.weight
        self._chk(self.log_sigma2, "log_sigma2(param)")
        self._chk(W, "W")
        logW = torch.log(W * W + 1e-8)
        self._chk(logW, "log(W²)")
        log_alpha = self._clip(self.log_sigma2 - logW)
        self._chk(log_alpha, "log_alpha")
        mdkl = (
            k1 * torch.sigmoid(k2 + k3 * log_alpha)
            - 0.5 * torch.log1p(torch.exp(-log_alpha))
            + C
        )
        self._chk(mdkl, "mdkl")
        if self.mode == "original":
            result = -mdkl.sum()
        else:
            result = -mdkl.mean()
        self._chk(result, "kl_loss_result")
        return result


def iter_vd_convs(module):
    """Yield (name, VDConv2d) for every VDConv2d in the module tree."""
    for name, child in module.named_modules():
        if isinstance(child, VDConv2d):
            yield name, child


def add_vd_to_conv(module, ard_init=-10.0, mode="channel", thresh=3.0, train_clip=False):
    """Recursively replace nn.Conv2d with VDConv2d (in-place).

    Args:
        module:     root module (e.g. ResNet) to walk.
        ard_init:   initial value for log_sigma2 (default -10).
        mode:       "channel" (per-channel sigma) or "original" (per-weight, exact paper).
        thresh:     log_alpha pruning threshold (default 3, only used in "original" mode).
        train_clip: zero high-log_alpha weights during training too (default False).

    Must be called BEFORE wrapping the model in SparseModel so that
    SparseModel._stat_layer_info() picks up the nested parameter names
    (e.g. 'conv1.conv.weight' instead of 'conv1.weight').
    """
    for name, child in module.named_children():
        if isinstance(child, VDConv2d):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, VDConv2d(
                child, ard_init=ard_init, mode=mode,
                thresh=thresh, train_clip=train_clip))
            continue
        add_vd_to_conv(child, ard_init=ard_init, mode=mode,
                        thresh=thresh, train_clip=train_clip)
    return module


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, KD=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        x = self.fc(x_f)  # B x num_classes
        if self.KD == True:
            return x_f, x
        else:
            return x

def resnet18(class_num, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], class_num, **kwargs)
    return model


def resnet56(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(Bottleneck, [6, 6, 6], class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


def resnet110(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [12, 12, 12], class_num, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
