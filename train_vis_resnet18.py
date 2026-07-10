import argparse, os, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
from torchvision.models import resnet18
import plotly.graph_objects as go

DATASETS = {
    'cifar10': {'cls': 10,  'mean': (0.4914,0.4822,0.4465), 'std': (0.2470,0.2435,0.2616)},
    'cifar100':{'cls': 100, 'mean': (0.5071,0.4865,0.4409), 'std': (0.2673,0.2564,0.2762)},
    'svhn':    {'cls': 10,  'mean': (0.4377,0.4438,0.4728), 'std': (0.1980,0.2010,0.1970)},
    'cinic10': {'cls': 10,  'mean': (0.4789,0.4723,0.4305), 'std': (0.2421,0.2383,0.2587)},
    'tinyimagenet':{'cls':200,'mean': (0.5,0.5,0.5),          'std': (0.5,0.5,0.5)},
}

def get_dataloader(ds_name, data_dir, batch_size):
    cfg = DATASETS[ds_name]
    root = os.path.join(data_dir, ds_name)
    sz = 64 if ds_name == 'tinyimagenet' else 32
    tf_tr = T.Compose([T.RandomCrop(sz, padding=4), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(cfg['mean'], cfg['std'])])
    tf_te = T.Compose([T.ToTensor(), T.Normalize(cfg['mean'], cfg['std'])])
    if ds_name == 'cifar10':
        tr = torchvision.datasets.CIFAR10(root, train=True,  download=True, transform=tf_tr)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf_te)
    elif ds_name == 'cifar100':
        tr = torchvision.datasets.CIFAR100(root, train=True,  download=True, transform=tf_tr)
        te = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=tf_te)
    elif ds_name == 'svhn':
        tr = torchvision.datasets.SVHN(root, split='train', download=True, transform=tf_tr)
        te = torchvision.datasets.SVHN(root, split='test',  download=True, transform=tf_te)
    else:
        tr = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform=tf_tr)
        te = torchvision.datasets.ImageFolder(os.path.join(root, 'val' if ds_name == 'tinyimagenet' else 'test'), transform=tf_te)
    train_loader = torch.utils.data.DataLoader(tr, batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(te, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader, cfg['cls']

def record_scores(model, loader, criterion, device):
    """Forward+backward on one batch, return {module_name: {mag, mag_grad}}."""
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    conv_linear = {n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}
    scores = {}
    for n in conv_linear:
        w = dict(model.named_parameters())[f'{n}.weight']
        if w.grad is None: continue
        w_np, g_np = w.data.detach().cpu().numpy().ravel(), w.grad.detach().cpu().numpy().ravel()
        scores[n] = {'mag': np.abs(w_np), 'mag_grad': np.abs(w_np) * np.abs(g_np)}
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10', choices=list(DATASETS))
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--num_bins', type=int, default=200, help='点数（分布图分箱数 / 质量曲线采样数）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--p', type=float, default=0.99, help='保留的质量比例（默认0.99=保留99%的权重总幅值）')
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    train_loader, test_loader, n_cls = get_dataloader(args.dataset, args.data_dir, args.batch_size)
    print(f'Dataset: {args.dataset}, classes: {n_cls}, batches/epoch: {len(train_loader)}')

    model = resnet18(num_classes=n_cls, weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.eval()
    with torch.no_grad(): model(next(iter(train_loader))[0].to(args.device))

    layers = sorted([n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
    print(f'Tracking {len(layers)} layers')

    NB = args.num_bins
    data = {n: {mk: {'bins':[], 'hist':[], 'mass':[]} for mk in ('mag','mag_grad')} for n in layers}
    mm = {n: {mk: {'min':np.inf,'max':-np.inf} for mk in ('mag','mag_grad')} for n in layers}

    # Common x-axis for the mass retention curve: fraction of weights kept (0 → 1)
    mass_x = np.linspace(0, 1, NB)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        scores = record_scores(model, train_loader, criterion, args.device)
        for n in layers:
            for mk in ('mag', 'mag_grad'):
                arr = scores[n][mk]
                # Global min/max (for histogram common grid)
                mm[n][mk]['min'] = min(mm[n][mk]['min'], float(arr.min()))
                mm[n][mk]['max'] = max(mm[n][mk]['max'], float(arr.max()))
                # Histogram with adaptive bins
                bins = np.linspace(arr.min(), arr.max(), NB + 1)
                ctr = (bins[:-1] + bins[1:]) / 2
                h, _ = np.histogram(arr, bins=bins)
                data[n][mk]['bins'].append(ctr)
                data[n][mk]['hist'].append(h)
                # Mass retention curve: sort desc → cumsum → normalize → resample to [0,1]
                sorted_desc = np.sort(arr)[::-1]
                cum = np.cumsum(sorted_desc) / np.sum(sorted_desc)
                x_orig = np.arange(1, len(arr) + 1) / len(arr)
                data[n][mk]['mass'].append(np.interp(mass_x, x_orig, cum, left=0))

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                p = model(x.to(args.device)).argmax(1)
                correct += (p == y.to(args.device)).sum().item(); total += y.size(0)
        acc = correct / total * 100
        best_acc = max(best_acc, acc)
        print(f'Epoch {epoch+1:3d}/{args.epochs}  Loss:{running_loss/len(train_loader):.3f}  Acc:{acc:.2f}%  LR:{scheduler.get_last_lr()[0]:.5f}')

    print(f'Best test accuracy: {best_acc:.2f}%')

    # ----- Post-process: interpolate histograms to global grid; mass curves already aligned -----
    hist_surf, mass_surf, centers_surf = {}, {}, {}
    for n in layers:
        hist_surf[n] = {}; mass_surf[n] = {}; centers_surf[n] = {}
        for mk in ('mag', 'mag_grad'):
            gmin, gmax = mm[n][mk]['min'], mm[n][mk]['max']
            if gmax - gmin < 1e-12: gmax = gmin + 1e-6
            gbins = np.linspace(gmin, gmax, NB)
            hgrid = [np.interp(gbins, data[n][mk]['bins'][e], data[n][mk]['hist'][e], left=0, right=0)
                     for e in range(args.epochs)]
            hist_surf[n][mk] = np.array(hgrid)
            mass_surf[n][mk]  = np.array(data[n][mk]['mass'])  # (epochs, NB), already aligned
            centers_surf[n][mk] = gbins

    # ----- Build Plotly figures -----
    epochs_arr = np.arange(args.epochs)
    traces_dist, traces_mass = [], []

    for li, n in enumerate(layers):
        for mk in ('mag', 'mag_grad'):
            label = '|W|' if mk == 'mag' else '|W|·|∇W|'
            cs, vis = ('Blues' if mk == 'mag' else 'Reds'), (li == 0)
            traces_dist.append(go.Surface(
                x=centers_surf[n][mk], y=epochs_arr, z=hist_surf[n][mk],
                name=label, visible=vis, colorscale=cs, opacity=0.85,
                hovertemplate=f'Epoch:%{{y}}<br>Score:%{{x:.4f}}<br>Freq:%{{z:.0f}}<br>{label}<extra></extra>'))
            traces_mass.append(go.Surface(
                x=mass_x, y=epochs_arr, z=mass_surf[n][mk],
                name=label, visible=vis, colorscale=cs,
                opacity=(0.9 if mk == 'mag' else 0.55),  # mag_grad more transparent so both visible
                hovertemplate=f'Keep %weights:%{{x:.1%}}<br>Epoch:%{{y}}<br>Mass retained:%{{z:.4f}}<br>{label}<extra></extra>'))

    def make_buttons(n_layers):
        return [dict(buttons=[
            dict(label=layers[i], method='restyle',
                 args=[{'visible': [False]*(2*i) + [True,True] + [False]*(2*(n_layers-1-i))}])
            for i in range(n_layers)], direction='down', showactive=True, x=0.1, y=1.15)]

    fig_dist = go.Figure(data=traces_dist)
    fig_dist.update_layout(title='Score Distribution (3D)',
        scene=dict(xaxis_title='Score', yaxis_title='Epoch', zaxis_title='Frequency'),
        updatemenus=make_buttons(len(layers)), height=600)

    fig_mass = go.Figure(data=traces_mass)
    # z=p plane: mass fraction threshold
    xx, yy = np.meshgrid(mass_x, [0, args.epochs - 1])
    fig_mass.add_trace(go.Surface(
        x=mass_x, y=yy[:,0], z=np.full_like(xx, args.p),
        name=f'p={args.p}', colorscale='Greens', opacity=0.25, showscale=False,
        hovertemplate=f'Keep %weights:%{{x:.1%}}<br>Epoch:%{{y}}<br>Mass retained(p={args.p})<extra></extra>'))
    fig_mass.update_layout(
        title=f'Magnitude Mass Retention — what % of weights capture {args.p:.0%} of total magnitude?',
        scene=dict(xaxis_title='% weights kept (sorted by score ↓)', yaxis_title='Epoch',
                   zaxis_title='% total magnitude mass retained'),
        updatemenus=make_buttons(len(layers)), height=600)

    # ----- Combine into one self-contained HTML -----
    html = '<html><head><meta charset="utf-8"></head><body>'
    html += '<h2>ResNet18 Score Distribution &amp; Mass Retention</h2>'
    html += f'<p>Dataset: {args.dataset} | Epochs: {args.epochs} | Best Acc: {best_acc:.2f}% | p={args.p}</p>'
    html += fig_dist.to_html(full_html=False, include_plotlyjs=True)
    html += '<hr>'
    html += fig_mass.to_html(full_html=False, include_plotlyjs=False)
    html += '</body></html>'

    out = f'vis_{args.dataset}.html'
    with open(out, 'w') as f: f.write(html)
    print(f'Saved {out}')

if __name__ == '__main__':
    main()
