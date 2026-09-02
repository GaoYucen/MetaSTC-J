from __future__ import annotations

import argparse
import copy
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import signal
from scipy import special as ss
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


@contextmanager
def _autocast(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "cuda":
        name = "cuda:0"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return device


def _load_flow(dataset: str) -> Tuple[np.ndarray, list, float]:
    load_start = time.perf_counter()
    if dataset == "beijing":
        path = "data/traffic_flow/1/20230306/part-00000_new.pkl"
        import pickle
        with open(path, "rb") as handle:
            rows = pickle.load(handle)
        ids = [row["id"] for row in rows]
        flow = np.asarray([row["flow"] for row in rows], dtype=np.float32)
        feature = pd.read_csv("data/link_feature.csv")
        feature_ids = set(feature["link_ID"].values)
        valid = np.asarray([rid in feature_ids for rid in ids])
        flow = flow[valid]
        ids = [rid for rid, keep in zip(ids, valid) if keep]
    elif dataset == "shanghai":
        path = "data/traffic_flow/4/20230602/shanghai_0602.pkl"
        import pickle
        with open(path, "rb") as handle:
            rows = pickle.load(handle)
        ids = [row["id"] for row in rows]
        flow = np.asarray([row["flow"] for row in rows], dtype=np.float32)
    elif dataset == "largest":
        cache_path = Path("data/gla/gla_his_2019_first288_paper_scale.npy")
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing LargeST cache {cache_path}; run tools/extract_largest_first288.py first")
        flow = np.load(cache_path, allow_pickle=False).astype(np.float32, copy=False)
        ids = list(range(flow.shape[0]))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if flow.ndim != 2 or flow.shape[1] != 288:
        raise ValueError(f"Expected [nodes, 288] flow data, got {flow.shape}")
    scale = float(np.max(flow))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Flow data has an invalid maximum")
    print(f"data_load_seconds={time.perf_counter() - load_start:.3f} nodes={flow.shape[0]} time_steps={flow.shape[1]}")
    return flow / scale, ids, scale


def _cluster_features(dataset: str, family: str, flow: np.ndarray, ids: list, k: int, seed: int) -> np.ndarray:
    cluster_start = time.perf_counter()
    look_back = 12
    dynamic = np.stack(
        [np.mean(flow[:, j * look_back:(j + 1) * look_back], axis=1) for j in range(12)],
        axis=1,
    )
    pieces = [dynamic]
    if family == "film" and dataset == "beijing":
        feature = pd.read_csv("data/link_feature.csv")
        feature = feature.set_index("link_ID").reindex(ids)
        static = feature.drop(columns=["Kind", "geometry"], errors="ignore")
        static = static.select_dtypes(include=[np.number]).fillna(0.0).to_numpy(dtype=np.float32)
        if static.shape[0] != flow.shape[0]:
            raise ValueError("Static features are not aligned with flow data")
        denom = np.max(np.abs(static), axis=0, keepdims=True)
        static = np.divide(static, np.where(denom > 0, denom, 1.0))
        pieces.insert(0, static)
    features = np.concatenate(pieces, axis=1)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
    print(f"cluster_seconds={time.perf_counter() - cluster_start:.3f} clusters={k}")
    return labels.astype(np.int64)


def _windows(flow: np.ndarray, look_back: int, look_forward: int):
    train_size = int(flow.shape[1] * 0.8)
    train_part = flow[:, :train_size].T
    test_part = flow[:, train_size:].T

    def build(part: np.ndarray):
        count = part.shape[0] - look_back - look_forward
        if count <= 0:
            return (
                np.empty((0, look_back, part.shape[1]), dtype=np.float32),
                np.empty((0, look_forward, part.shape[1]), dtype=np.float32),
            )
        x = np.stack([part[i:i + look_back] for i in range(count)]).astype(np.float32)
        y = np.stack([part[i + look_back:i + look_back + look_forward] for i in range(count)]).astype(np.float32)
        return x, y

    train_x, train_y = build(train_part)
    test_x, test_y = build(test_part)
    split = max(1, int(train_x.shape[0] * 0.9))
    return train_x[:split], train_y[:split], train_x[split:], train_y[split:], test_x, test_y


def _by_cluster(x: np.ndarray, y: np.ndarray, labels: np.ndarray, k: int, look_back: int, look_forward: int):
    result = {}
    for cluster in range(k):
        roads = np.flatnonzero(labels == cluster)
        if roads.size == 0:
            raise ValueError(f"Cluster {cluster} is empty")
        cx = x[:, :, roads].transpose(2, 0, 1).reshape(-1, look_back, 1)
        cy = y[:, :, roads].transpose(2, 0, 1).reshape(-1, look_forward, 1)
        result[cluster] = (torch.from_numpy(cx), torch.from_numpy(cy))
    return result


def _loaders(data: Dict[int, Tuple[torch.Tensor, torch.Tensor]], batch_size: int, device: torch.device, shuffle: bool):
    return {
        cluster: DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for cluster, (x, y) in data.items()
    }


class LSTMModel(nn.Module):
    def __init__(self, look_back: int, look_forward: int, hidden_dim: int = 20):
        super().__init__()
        self.look_forward = look_forward
        self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
        self.linear = nn.Linear(look_back * hidden_dim, look_forward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        output = self.linear(output.reshape(output.shape[0], -1))
        return output.reshape(output.shape[0], self.look_forward, 1)


def transition(order: int):
    q = np.arange(order, dtype=np.float64)
    r = (2 * q + 1)[:, None]
    j, i = np.meshgrid(q, q)
    a = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * r
    b = (-1.0) ** q[:, None] * r
    return a, b


@torch.jit.script
def hippo_recurrence(inputs_p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c_init: torch.Tensor):
    length = inputs_p.size(0)
    batch = inputs_p.size(1)
    channels = inputs_p.size(2)
    order = a.size(0)
    c = c_init
    states = torch.empty((length, batch, channels, order), device=inputs_p.device, dtype=inputs_p.dtype)
    b_expanded = b.unsqueeze(0)
    at = a.t()
    for step in range(length):
        value = inputs_p[step].unsqueeze(-1)
        c = torch.matmul(c, at) + value @ b_expanded
        states[step] = c
    return states


class HiPPO_LegT(nn.Module):
    def __init__(self, order: int, dt: float):
        super().__init__()
        a, b = transition(order)
        c = np.ones((1, order))
        d = np.zeros((1,))
        a, b, _, _, _ = signal.cont2discrete((a, b, c, d), dt=dt, method="bilinear")
        self.order = order
        self.register_buffer("a", torch.tensor(a, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(b.squeeze(-1), dtype=torch.float32))
        values = np.arange(0.0, 1.0, dt)
        matrix = ss.eval_legendre(np.arange(order)[:, None], 1 - 2 * values).T
        self.register_buffer("eval_matrix", torch.tensor(matrix, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        c_init = torch.zeros(inputs.shape[0], inputs.shape[1], self.order, device=inputs.device, dtype=inputs.dtype)
        return hippo_recurrence(inputs.permute(2, 0, 1), self.a.to(inputs.dtype), self.b.to(inputs.dtype), c_init)


class SpectralConv1d(nn.Module):
    def __init__(self, channels: int, seq_len: int):
        super().__init__()
        self.channels = channels
        self.modes = max(1, min(32, seq_len // 2))
        scale = 1.0 / (channels * channels)
        self.weights_real = nn.Parameter(scale * torch.rand(channels, channels, self.modes))
        self.weights_imag = nn.Parameter(scale * torch.rand(channels, channels, self.modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, length = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            x.shape[0], x.shape[1], self.channels, length // 2 + 1,
            device=x.device, dtype=torch.cfloat,
        )
        modes = min(self.modes, x_ft.shape[-1])
        real = self.weights_real[:, :, :modes]
        imag = self.weights_imag[:, :, :modes]
        a = x_ft[:, :, :, :modes]
        out_ft[:, :, :, :modes] = torch.complex(
            torch.einsum("bjix,iox->bjox", a.real, real) - torch.einsum("bjix,iox->bjox", a.imag, imag),
            torch.einsum("bjix,iox->bjox", a.real, imag) + torch.einsum("bjix,iox->bjox", a.imag, real),
        )
        return torch.fft.irfft(out_ft, n=length, dim=-1)


class FiLMModel(nn.Module):
    def __init__(self, look_back: int, look_forward: int):
        super().__init__()
        self.look_back = look_back
        self.look_forward = look_forward
        self.multiscale = (1, 2, 4)
        self.window_size = (256,)
        self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))
        self.legts = nn.ModuleList([
            HiPPO_LegT(order=n, dt=1.0 / look_forward / scale)
            for n in self.window_size for scale in self.multiscale
        ])
        self.spectral = nn.ModuleList([
            SpectralConv1d(n, min(look_forward, look_back))
            for n in self.window_size for _ in self.multiscale
        ])
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        normalized = centered / stdev
        normalized = normalized * self.affine_weight + self.affine_bias
        decoded = []
        for index, scale in enumerate(self.multiscale):
            input_len = scale * self.look_forward
            x_in = normalized[:, -input_len:]
            legt = self.legts[index]
            coeff = legt(x_in.transpose(1, 2)).permute(1, 2, 3, 0)
            spectral = self.spectral[index](coeff)
            time_index = min(self.look_forward - 1, spectral.shape[-1] - 1)
            coeff_at_horizon = spectral.transpose(2, 3)[:, :, time_index, :]
            decoded.append(coeff_at_horizon @ legt.eval_matrix.to(x_enc.dtype)[-self.look_forward:, :].T)
        output = torch.stack(decoded, dim=-1)
        output = self.mlp(output).squeeze(-1).permute(0, 2, 1)
        output = output - self.affine_bias
        output = output / (self.affine_weight + 1e-10)
        return output * stdev + means

    def forward(self, x_enc: torch.Tensor, *args) -> torch.Tensor:
        return self.forecast(x_enc)


class FirstOrderTrainer:
    def __init__(self, model: nn.Module, family: str, device: torch.device, amp: bool):
        self.model = model
        self.family = family
        self.device = device
        self.amp = amp
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
        self.loss_fn = nn.MSELoss()

    def inner_step(self, data: torch.Tensor, targets: torch.Tensor, lr: float = 0.01) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.amp and self.family == "lstm", self.device):
            predictions = self.model(data)
            loss = self.loss_fn(predictions, targets)
        gradients = torch.autograd.grad(loss, tuple(self.model.parameters()), create_graph=False)
        with torch.no_grad():
            for parameter, gradient in zip(self.model.parameters(), gradients):
                parameter.add_(gradient, alpha=-lr)
        return float(loss.detach().cpu())

    def outer_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        self.inner_step(data, targets)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast(self.amp and self.family == "lstm", self.device):
            predictions = self.model(data)
            loss = self.loss_fn(predictions, targets)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        return float(loss.detach().cpu())


def _move(batch, device):
    data, targets = batch
    return data.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def _limited(loader: Iterable, max_batches: Optional[int]):
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        yield batch


def _validation(model: nn.Module, loaders: Dict[int, DataLoader], device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for loader in loaders.values():
            for batch in loader:
                data, targets = _move(batch, device)
                losses.append(float(nn.functional.mse_loss(model(data), targets).cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    mask = targets != 0
    mape = float(np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask]))) if np.any(mask) else 0.0
    return {
        "MAE": float(mean_absolute_error(targets, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(targets, predictions))),
        "MSE": float(mean_squared_error(targets, predictions)),
        "MAPE": mape,
        "R2": float(r2_score(targets, predictions)),
    }


def _plot(path: Path, predictions: np.ndarray, targets: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    length = min(300, predictions.size)
    figure = plt.figure(figsize=(12, 5))
    plt.plot(targets[:length], label="Targets", color="grey")
    plt.plot(predictions[:length], label="Predictions", color="red")
    plt.legend()
    plt.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run_experiment(defaults: dict) -> None:
    parser = argparse.ArgumentParser(description="MetaSTC-J 4090 optimized experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--adapt-epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.epochs <= 0 or args.adapt_epochs <= 0:
        raise ValueError("batch-size, epochs, and adapt-epochs must be positive")
    device = _device(args.device)
    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    family = defaults["family"]
    dataset = defaults["dataset"]
    look_back = 12
    look_forward = 6
    k = defaults["clusters"]
    amp = family == "lstm" and device.type == "cuda"

    flow, ids, scale = _load_flow(dataset)
    labels = _cluster_features(dataset, family, flow, ids, k, args.seed)
    prep_start = time.perf_counter()
    np.savetxt(output_dir / "cluster_labels.txt", labels, fmt="%d")
    fit_x, fit_y, val_x, val_y, test_x, test_y = _windows(flow, look_back, look_forward)
    fit = _by_cluster(fit_x, fit_y, labels, k, look_back, look_forward)
    validation = _by_cluster(val_x, val_y, labels, k, look_back, look_forward)
    testing = _by_cluster(test_x, test_y, labels, k, look_back, look_forward)
    train_loaders = _loaders(fit, args.batch_size, device, True)
    val_loaders = _loaders(validation, args.batch_size, device, False)
    test_loaders = _loaders(testing, args.batch_size, device, False)
    cluster_counts = np.bincount(labels, minlength=k).tolist()
    print(f"data_prep_seconds={time.perf_counter() - prep_start:.3f} cluster_counts={cluster_counts}")

    model = LSTMModel(look_back, look_forward) if family == "lstm" else FiLMModel(look_back, look_forward)
    model = model.to(device)
    if family == "lstm":
        model.lstm.flatten_parameters()
    trainer = FirstOrderTrainer(model, family, device, amp)
    global_path = output_dir / "global_best.pt"
    config = {
        "family": family,
        "dataset": dataset,
        "look_back": look_back,
        "look_forward": look_forward,
        "clusters": k,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "adapt_epochs": args.adapt_epochs,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "device": str(device),
        "amp": amp,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))
    print(f"dataset_shape={tuple(flow.shape)} train_samples={fit_x.shape[0]} val_samples={val_x.shape[0]} test_samples={test_x.shape[0]} train_batches={sum(len(loader) for loader in train_loaders.values())}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    log_path = output_dir / "train.log"
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    start = time.time()
    with log_path.open("w") as log:
        for epoch in range(args.epochs):
            losses = []
            for cluster in range(k):
                for batch in _limited(train_loaders[cluster], args.max_batches):
                    data, targets = _move(batch, device)
                    losses.append(trainer.outer_step(data, targets))
            validation_loss = _validation(model, val_loaders, device)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, global_path)
            message = f"epoch={epoch + 1}/{args.epochs} train_loss={np.mean(losses):.7f} val_loss={validation_loss:.7f} elapsed={time.time() - start:.2f}s"
            print(message)
            log.write(message + "\n")
    model.load_state_dict(best_state)

    for cluster in range(k):
        model.load_state_dict(best_state)
        adapter = FirstOrderTrainer(model, family, device, amp=False)
        for _ in range(args.adapt_epochs):
            for batch in _limited(train_loaders[cluster], args.max_batches):
                data, targets = _move(batch, device)
                adapter.inner_step(data, targets)
        torch.save(model.state_dict(), output_dir / f"cluster_{cluster}.pt")

    predictions = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for cluster in range(k):
            model.load_state_dict(torch.load(output_dir / f"cluster_{cluster}.pt", map_location=device, weights_only=True))
            for batch in test_loaders[cluster]:
                data, target = _move(batch, device)
                predictions.append((model(data) * scale).cpu().numpy())
                targets.append((target * scale).cpu().numpy())
    pred_array = np.concatenate(predictions).reshape(-1)
    target_array = np.concatenate(targets).reshape(-1)
    metrics = _metrics(pred_array, target_array)
    metrics["train_seconds"] = round(time.time() - start, 3)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with log_path.open("a") as log:
        log.write(json.dumps(metrics) + "\n")
    print(json.dumps(metrics, indent=2))
    if not args.no_plot:
        _plot(output_dir / "prediction.png", pred_array, target_array)
